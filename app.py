import os
import re
import secrets
import uuid

from flask import Flask, render_template, request
from flask import flash, redirect, session, url_for
import pandas as pd
from PIL import Image, ImageEnhance, UnidentifiedImageError
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import pytesseract
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

mongo_client = MongoClient("mongodb://localhost:27017/")
database = mongo_client["dermascan_db"]

existing_collections = database.list_collection_names()
if "skin_profiles" not in existing_collections:
    database.create_collection("skin_profiles")
if "users" not in existing_collections:
    database.create_collection("users")

skin_profiles_collection = database["skin_profiles"]
users_collection = database["users"]
users_collection.create_index("email", unique=True)

ALLOWED_IMAGE_FORMATS = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
}

ingredients_df = pd.read_csv(
    os.path.join(app.root_path, "data", "ingredients.csv")
)

def extract_ingredient_list_text(ocr_text):
    ingredient_heading = re.search(
        r"(?im)\bingredients?\b[ \t]*(?:[:：\-–—][ \t]*|(?=\r?$))",
        ocr_text,
    )
    if ingredient_heading is None:
        return ""

    ingredient_text = ocr_text[ingredient_heading.end():]
    next_section = re.search(
        r"(?i)(?<![a-z0-9])(?:directions?|how\s+to\s+use|warnings?|"
        r"cautions?|storage|distributed\s+by|manufactured\s+by|made\s+in|"
        r"net\s+contents|net\s+wt|website)\b[ \t]*[:\-]?",
        ingredient_text,
    )
    if next_section is not None:
        ingredient_text = ingredient_text[:next_section.start()]

    return ingredient_text.strip()


def find_detected_ingredients(ocr_text, debug_matches=None):
    def normalize_text(value):
        value = str(value).casefold()
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    ingredient_list_text = extract_ingredient_list_text(ocr_text)
    normalized_ocr = normalize_text(ingredient_list_text)
    if not normalized_ocr:
        return []

    matches = []
    for _, ingredient in ingredients_df.iterrows():
        display_name = ingredient["common_name"]
        if pd.isna(display_name) or not str(display_name).strip():
            display_name = ingredient["inci_name"]
        display_name = str(display_name).strip()

        ingredient_names = [(ingredient["inci_name"], "INCI")]
        aliases = ingredient["also_known_as"]
        if pd.notna(aliases):
            ingredient_names.extend(
                (alias, "alias") for alias in str(aliases).split(";")
            )

        for name, matching_method in ingredient_names:
            if pd.isna(name):
                continue

            normalized_name = normalize_text(name)
            if not normalized_name:
                continue
            if (
                matching_method == "alias"
                and len(normalized_name.replace(" ", "")) < 4
            ):
                continue

            name_pattern = r"\s+".join(
                re.escape(token) for token in normalized_name.split()
            )
            for match in re.finditer(
                rf"(?<![a-z0-9]){name_pattern}(?![a-z0-9])",
                normalized_ocr,
            ):
                matches.append({
                    "start": match.start(),
                    "end": match.end(),
                    "matched_phrase": match.group(),
                    "name": display_name,
                    "ingredient_id": ingredient["id"],
                    "matching_method": matching_method,
                    "length": match.end() - match.start(),
                })

    matches = [
        match
        for match in matches
        if not any(
            other["ingredient_id"] != match["ingredient_id"]
            and other["length"] > match["length"]
            and other["start"] <= match["start"]
            and other["end"] >= match["end"]
            for other in matches
        )
    ]
    matches.sort(
        key=lambda match: (
            match["start"],
            -match["length"],
            match["matching_method"] != "INCI",
            match["name"].casefold(),
        )
    )

    detected_ingredients = []
    detected_names = set()
    for match in matches:
        if match["name"].casefold() in detected_names:
            continue

        detected_ingredients.append(match["name"])
        detected_names.add(match["name"].casefold())
        if debug_matches is not None:
            debug_matches.append({
                "ocr_text": ocr_text,
                "matched_phrase": match["matched_phrase"],
                "dataset_ingredient_name": match["name"],
                "match_source": match["matching_method"],
                "matching_method": "exact normalized phrase",
            })

    return detected_ingredients

def get_ingredient_information(detected_ingredients):
    ingredient_information = []
    information_fields = {
        "Function": "category",
        "Benefits": "benefits",
        "Skin types": "skin_types",
        "Cautions": "concerns",
        "Irritancy": "irritancy",
        "Safety": "pregnancy_safe",
    }

    for ingredient_name in detected_ingredients:
        matching_rows = ingredients_df[
            ingredients_df["common_name"].astype(str).str.casefold()
            == ingredient_name.casefold()
        ]
        if matching_rows.empty:
            matching_rows = ingredients_df[
                ingredients_df["inci_name"].astype(str).str.casefold()
                == ingredient_name.casefold()
            ]

        if matching_rows.empty:
            ingredient_information.append({
                "name": ingredient_name,
                "details": {
                    label: "Information not available."
                    for label in information_fields
                },
            })
            continue

        ingredient = matching_rows.iloc[0]
        details = {}
        for label, column in information_fields.items():
            value = ingredient[column]
            if pd.isna(value) or not str(value).strip():
                details[label] = "Information not available."
            else:
                details[label] = str(value).strip()

        ingredient_information.append({
            "name": ingredient_name,
            "details": details,
        })

    return ingredient_information

def get_product_compatibility(detected_ingredients, skin_profile):
    if not detected_ingredients or not skin_profile:
        return None

    concern_terms = {
        "acne": ("acne", "blemish", "pore"),
        "dryness": ("hydrat", "moistur", "dry"),
        "pigmentation": (
            "pigmentation", "dark spot", "brighten", "even tone"
        ),
        "aging": (
            "aging", "fine line", "wrinkle", "mature", "expression line",
            "elasticity",
        ),
        "redness": (
            "redness", "soothe", "calm", "irritation", "rosacea"
        ),
    }
    sensitive_caution_terms = (
        "allergy", "sensitivity", "sensit", "irritat", "dryness",
        "drying", "purging", "tingling", "sting",
    )
    profile_caution_terms = {
        "acne": (
            "comedogenic", "clogged pores", "clog pores", "acne-prone",
            "acne prone",
        ),
        "dryness": ("drying", "stripping", "over-dry", "over dry"),
        "pigmentation": ("photosensit", "sun sensitivity"),
        "redness": (
            "allergy", "sensitivity", "sensit", "irritat", "sting",
            "tingling",
        ),
    }

    skin_type = str(skin_profile.get("skin_type", "")).strip().casefold()
    skin_concern = str(skin_profile.get("skin_concern", "")).strip().casefold()
    sensitive_skin = (
        str(skin_profile.get("sensitive_skin", "")).strip().casefold() == "yes"
    )
    concern_keywords = concern_terms.get(skin_concern, ())
    ingredient_summary = []
    product_reasons = []
    type_matches = 0
    unknown_skin_type_fit = 0
    concern_matches = 0
    ingredient_cautions = 0
    ingredient_scores = []

    for ingredient_name in detected_ingredients:
        matching_rows = ingredients_df[
            ingredients_df["common_name"].astype(str).str.casefold()
            == ingredient_name.casefold()
        ]
        if matching_rows.empty:
            matching_rows = ingredients_df[
                ingredients_df["inci_name"].astype(str).str.casefold()
                == ingredient_name.casefold()
            ]

        if matching_rows.empty:
            ingredient_summary.append({
                "name": ingredient_name,
                "status": "Insufficient Data / Unknown",
                "reason": "Ingredient information is unavailable in the dataset.",
            })
            ingredient_scores.append(50)
            unknown_skin_type_fit += 1
            continue

        ingredient = matching_rows.iloc[0]
        score = 50
        reasons = []
        has_caution = False
        listed_skin_types = {
            value.strip().casefold()
            for value in str(ingredient.get("skin_types", "")).split(";")
            if value.strip() and value.strip().casefold() != "nan"
        }
        skin_type_match = (
            "all" in listed_skin_types or skin_type in listed_skin_types
        )

        if listed_skin_types:
            if skin_type_match:
                score += 25
                type_matches += 1
                reasons.append("Dataset lists your skin type as a match (+25).")
            else:
                unknown_skin_type_fit += 1
                reasons.append(
                    "Your skin type is not specified for this ingredient; no compatibility penalty applied (0)."
                )
        else:
            unknown_skin_type_fit += 1
            reasons.append("No skin-type fit is listed in the dataset.")

        benefits = str(ingredient.get("benefits", "")).casefold()
        if any(keyword in benefits for keyword in concern_keywords):
            score += 15
            concern_matches += 1
            reasons.append(
                "Listed benefits mention your selected concern (+15)."
            )

        irritancy = str(ingredient.get("irritancy", "")).strip().casefold()
        if sensitive_skin:
            if irritancy == "low":
                score += 5
                reasons.append("Dataset rates irritancy low for sensitive skin (+5).")
            elif irritancy == "medium":
                score -= 15
                has_caution = True
                reasons.append(
                    "Dataset rates irritancy medium for sensitive skin (-15)."
                )
            elif irritancy == "high":
                score -= 30
                has_caution = True
                reasons.append(
                    "Dataset rates irritancy high for sensitive skin (-30)."
                )
        elif irritancy == "medium":
            score -= 10
            has_caution = True
            reasons.append("Dataset rates irritancy medium (-10).")
        elif irritancy == "high":
            score -= 20
            has_caution = True
            reasons.append("Dataset rates irritancy high (-20).")

        caution_text = str(ingredient.get("concerns", "")).casefold()
        relevant_caution_terms = set(profile_caution_terms.get(skin_concern, ()))
        if sensitive_skin or skin_type == "sensitive":
            relevant_caution_terms.update(sensitive_caution_terms)

        matched_cautions = [
            term for term in relevant_caution_terms if term in caution_text
        ]
        if matched_cautions:
            score -= 10
            has_caution = True
            reasons.append(
                "Dataset warning is relevant to your skin profile "
                f"({', '.join(sorted(matched_cautions))}; -10)."
            )
        if irritancy in {"medium", "high"}:
            has_caution = True
        if has_caution:
            ingredient_cautions += 1

        score = max(0, min(100, score))
        ingredient_scores.append(score)
        if score < 50:
            status = "Not Recommended"
        elif has_caution:
            status = "Use with Caution"
        elif not skin_type_match:
            status = "Insufficient Data / Unknown"
        elif score < 75:
            status = "Use with Caution"
        else:
            status = "Compatible"

        if not reasons:
            reasons.append("No profile-relevant compatibility data is available.")

        ingredient_summary.append({
            "name": ingredient_name,
            "status": status,
            "reason": " ".join(reasons),
        })

    score = round(sum(ingredient_scores) / len(ingredient_scores))
    if score >= 75:
        label = "Recommended"
        icon = "🟢"
    elif score >= 50:
        label = "Use with Caution"
        icon = "🟡"
    else:
        label = "Not Recommended"
        icon = "🔴"

    if type_matches:
        product_reasons.append(
            f"{type_matches} detected ingredient(s) list your skin type as a match."
        )
    if unknown_skin_type_fit:
        product_reasons.append(
            f"Skin-type compatibility is unknown for {unknown_skin_type_fit} detected ingredient(s); missing matches were not penalized."
        )
    if concern_matches:
        product_reasons.append(
            f"{concern_matches} detected ingredient(s) have benefits mentioning your skin concern."
        )
    elif concern_keywords:
        product_reasons.append(
            "The available benefits data does not specifically mention your selected concern."
        )
    if ingredient_cautions:
        product_reasons.append(
            f"{ingredient_cautions} profile-relevant irritancy rating(s) or dataset warning(s) contributed caution."
        )
    if not product_reasons:
        product_reasons.append(
            "The available ingredient data provides no specific skin-type or concern match."
        )

    return {
        "score": score,
        "label": label,
        "icon": icon,
        "reasons": product_reasons,
        "ingredient_summary": ingredient_summary,
    }

def extract_text_from_image(image_path):
    with Image.open(image_path) as image:
        grayscale_image = image.convert("L")

    upscaled_image = grayscale_image.resize(
        (grayscale_image.width * 3, grayscale_image.height * 3),
        Image.Resampling.LANCZOS,
    )
    high_contrast_image = ImageEnhance.Contrast(upscaled_image).enhance(2.0)

    ocr_results = []
    for threshold in (160, 190):
        binary_image = high_contrast_image.point(
            lambda pixel: 255 if pixel > threshold else 0
        )
        for page_segmentation_mode in (6, 11):
            config = f"--oem 3 --psm {page_segmentation_mode}"
            ocr_results.append(
                pytesseract.image_to_string(binary_image, config=config)
            )

    combined_lines = []
    seen_lines = set()
    for ocr_result in ocr_results:
        for line in ocr_result.splitlines():
            line = line.strip()
            normalized_line = re.sub(r"[^a-z0-9]+", "", line.casefold())
            if normalized_line and normalized_line not in seen_lines:
                combined_lines.append(line)
                seen_lines.add(normalized_line)

    return "\n".join(combined_lines).strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password.strip():
            return render_template(
                "signup.html",
                error="Please provide your name, email, and password.",
            ), 400

        if users_collection.find_one({"email": email}):
            return render_template(
                "signup.html",
                error="An account with that email already exists.",
            ), 409

        user = {
            "name": name,
            "email": email,
            "password": generate_password_hash(password),
        }
        try:
            users_collection.insert_one(user)
        except DuplicateKeyError:
            return render_template(
                "signup.html",
                error="An account with that email already exists.",
            ), 409

        flash("Your account has been created. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = users_collection.find_one({"email": email}) if email else None

        if not user or not check_password_hash(user["password"], password):
            return render_template(
                "login.html",
                error="Invalid email or password.",
            ), 401

        session.clear()
        session["user_id"] = str(user["_id"])
        session["user_name"] = user["name"]
        return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    skin_profile = skin_profiles_collection.find_one(
        {"user_id": session["user_id"]},
        sort=[("_id", -1)],
    )
    return render_template(
        "dashboard.html",
        user_name=session.get("user_name"),
        skin_profile=skin_profile,
    )

@app.route("/upload-product", methods=["GET", "POST"])
def upload_product():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        image = request.files.get("product_image")
        if image is None or not image.filename:
            return render_template(
                "product_upload.html",
                error="Please select a product image to upload.",
            ), 400

        safe_filename = secure_filename(image.filename)
        extension = safe_filename.rsplit(".", 1)[-1].lower() if "." in safe_filename else ""
        expected_format = ALLOWED_IMAGE_FORMATS.get(extension)
        if expected_format is None:
            return render_template(
                "product_upload.html",
                error="Please upload a JPG, JPEG, or PNG image.",
            ), 400

        try:
            with Image.open(image.stream) as uploaded_image:
                if uploaded_image.format != expected_format:
                    return render_template(
                        "product_upload.html",
                        error="The selected file does not match its image extension.",
                    ), 400
                uploaded_image.verify()
            image.stream.seek(0)
        except (UnidentifiedImageError, OSError):
            return render_template(
                "product_upload.html",
                error="The selected file is not a valid image.",
            ), 400

        saved_filename = f"{uuid.uuid4().hex}.{extension}"
        saved_image_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
        image.save(saved_image_path)

        try:
            extracted_text = extract_text_from_image(saved_image_path)
        except pytesseract.TesseractNotFoundError:
            return render_template(
                "ocr_result.html",
                filename=safe_filename,
                error="The OCR engine is unavailable. The image was uploaded, but text could not be read.",
            ), 500
        except pytesseract.TesseractError:
            return render_template(
                "ocr_result.html",
                filename=safe_filename,
                error="OCR could not process this image. The image was uploaded successfully.",
            ), 500

        ingredient_match_debug = []
        detected_ingredients = find_detected_ingredients(
            extracted_text,
            debug_matches=ingredient_match_debug,
        )
        skin_profile = skin_profiles_collection.find_one(
            {"user_id": session["user_id"]},
            sort=[("_id", -1)],
        )
        return render_template(
            "ocr_result.html",
            filename=safe_filename,
            extracted_text=extracted_text,
            detected_ingredients=detected_ingredients,
            ingredient_information=get_ingredient_information(
                detected_ingredients
            ),
            skin_profile=skin_profile,
            compatibility=get_product_compatibility(
                detected_ingredients, skin_profile
            ),
        )

    return render_template("product_upload.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        skin_type = request.form.get("skin_type", "").strip()
        skin_concern = request.form.get("skin_concern", "").strip()
        sensitive_skin = request.form.get("sensitive_skin", "").strip()

        valid_skin_types = {"oily", "dry", "combination", "normal", "sensitive"}
        valid_skin_concerns = {
            "acne", "dryness", "pigmentation", "aging", "redness", "none"
        }
        valid_sensitive_skin = {"yes", "no"}

        if (
            skin_type not in valid_skin_types
            or skin_concern not in valid_skin_concerns
            or sensitive_skin not in valid_sensitive_skin
        ):
            return render_template(
                "profile.html",
                error="Please provide a valid choice for each profile field.",
            ), 400

        skin_profiles_collection.insert_one({
            "user_id": session["user_id"],
            "skin_type": skin_type,
            "skin_concern": skin_concern,
            "sensitive_skin": sensitive_skin,
        })
        return redirect(url_for("upload_product"))

    return render_template("profile.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)