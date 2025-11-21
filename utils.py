# utils.py
import os
import uuid
import json
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import fitz

# Configure the Gemini API key
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except AttributeError as e:
    print(f"Error: GOOGLE_API_KEY not found. Please set it in your .env file. {e}")
    exit()


# --- Helper function to merge two OCR results ---
def merge_ocr_results(primary_result, secondary_result):
    """
    Intelligently merges two JSON OCR results.
    Prefers the primary_result in case of conflicts.
    """
    if not isinstance(primary_result, dict) or not isinstance(secondary_result, dict):
        return primary_result or secondary_result

    merged_result = {}
    all_keys = set(primary_result.keys()) | set(secondary_result.keys())

    for key in all_keys:
        primary_value = primary_result.get(key)
        secondary_value = secondary_result.get(key)

        # Rule 1: If primary has a solid value, prefer it.
        if primary_value not in [None, "", []]:
            merged_result[key] = primary_value
        # Rule 2: If primary is empty but secondary has a value, use secondary.
        elif secondary_value not in [None, "", []]:
            merged_result[key] = secondary_value
        # Both are empty, just use primary's value (e.g., null)
        else:
            merged_result[key] = primary_value

    print(">>> OCR MERGE: Successfully merged two OCR results.")
    return merged_result


# --- Internal helper for single API call ---
def _call_gemini_api_single(file_path):
    """Internal function to perform a single OCR extraction."""
    try:
        # UPDATED: Changed model from 'gemini-1.5-flash' to 'gemini-2.0-flash'
        # as 1.5-flash is deprecated/retired for v1beta API users.
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        uploaded_file = genai.upload_file(path=file_path)

        prompt = """
        Analyze the provided image(s) of a Philippine Driver's License. Extract, parse, and enrich the information into a single, structured JSON object.

        **CRITICAL INSTRUCTIONS:**
        1.  **JSON ONLY:** The entire output MUST be a single, valid JSON object. Do not wrap it in markdown. Your response must start with `{` and end with `}`.
        2.  **PARSE THE NAME:** From the full name (e.g., "TORRALBA, ROLDAN UNNE CASTRO"), extract the `lastName`, `firstName`, and `middleName`.
            - The `lastName` is the part before the comma.
            - The `firstName` is the first word after the comma and space.
            - The `middleName` is **all remaining text** on that line after the first name. This can include multiple words (like "UNNE CASTRO"). If no text remains, the value should be `null`.
        3.  **IMPROVE ADDRESS ACCURACY:** Addresses in the Philippines often contain abbreviations. Pay close attention to words like `SUBD` (for Subdivision), `ST` (for Street), and `BRGY` (for Barangay). If a word is ambiguous but resembles one of these, prioritize the correct abbreviation. For example, if you see "SUED", correct it to "SUBD".
        4.  **MAP DL CODES:** Based on the extracted `dlCodes`, create a `dlCodesDetails` object by mapping each code to its full description from the provided list.
        5.  **HANDLE CONDITIONS:**
            - For the `conditions` field, extract the exact value from the card (e.g., "1", "4", "A,B,C,D,E"). If no conditions are listed, this field MUST be the string "NONE".
            - For the `conditionsDetails` field, this MUST ALWAYS be an array containing the full text of all five possible conditions, as listed below. This serves as a static reference list in every output.
        6.  **EXTRACT ORGAN DONATION:** For the `organDonation` field, you must extract the complete sentence written under that section (e.g., "I WILL NOT DONATE ANY ORGAN"). Do not summarize. If the section is blank or unreadable, the value should be `null`.
        7.  **ADHERE TO SCHEMA:** Use the exact field names and data types specified below. If a field is not present or unreadable, its value must be `null`.

        **FIELD-MAPPING DICTIONARIES (FOR YOUR REFERENCE):**
        *   **DL Codes Details Mapping:**
            - "A": "MOTORCYCLE", "A1": "TRICYCLE", "B": "UP TO 5000 KGS GVW/8 SEATS", "B1": "UP TO 5000 KGS GVW/9 OR MORE SEATS", "B2": "GOODS < 3500 KGS GVW", "BE": "TRAILERS < 3500 KGS", "C": "GOODS > 3500 KGS GVW", "CE": "ARTICULATED > 3500 KGS COMBINED GVW", "D": "BUS > 5000 KGS GVW/9 OR MORE SEATS"
        *   **Conditions Details Mapping (Use this to populate the static list):**
            - "A / 1": "WEAR CORRECTIVE LENSES", "B / 2": "DRIVE ONLY W/SPECIAL EQPT FOR UPPER/LOWER LIMBS", "C / 3": "DRIVE CUSTOMIZED MOTOR VEHICLE ONLY", "D / 4": "DAYLIGHT DRIVING ONLY", "E / 5": "HEARING AID REQUIRED"

        **REQUIRED JSON OUTPUT SCHEMA:**
        { "fullName": "string", "lastName": "string", "firstName": "string", "middleName": "string_or_null", "nationality": "string", "sex": "string", "dateOfBirth": "string (YYYY/MM/DD)", "weight": "integer", "height": "float", "address": "string", "licenseNumber": "string", "expirationDate": "string (YYYY/MM/DD)", "agencyCode": "string", "bloodType": "string", "eyesColor": "string", "dlCodes": ["array_of_strings"], "dlCodesDetails": { "object_key_value_pairs" }, "conditions": "string (The value from the card, e.g., '1' or 'NONE')", "conditionsDetails": ["WEAR CORRECTIVE LENSES", "DRIVE ONLY W/SPECIAL EQPT FOR UPPER/LOWER LIMBS", "DRIVE CUSTOMIZED MOTOR VEHICLE ONLY", "DAYLIGHT DRIVING ONLY", "HEARING AID REQUIRED"], "serialNumber": "integer", "emergencyContactName": "string", "emergencyContactAddress": "string", "emergencyContactTel": "string", "organDonation": "string (e.g., 'I WILL NOT DONATE ANY ORGAN' or null)" }
        """

        response = model.generate_content(
            [prompt, uploaded_file],
            safety_settings={'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                             'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                             'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
        )

        if not response.text.strip():
            print(f"!!! GEMINI ERROR on {os.path.basename(file_path)}: Received an empty response.")
            return {"error": "API returned an empty response, likely blocked by safety filters."}

        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned_text)
        return data

    except json.JSONDecodeError as e:
        print(f"!!! JSON DECODE ERROR on {os.path.basename(file_path)}: {e}")
        return {"error": "Failed to parse non-JSON response from API.", "raw_response": response.text}
    except Exception as e:
        print(f"!!! An unexpected error occurred calling Gemini API on {os.path.basename(file_path)}: {e}")
        return {"error": str(e)}


# --- The main OCR function, now with dual-pass logic ---
def perform_dual_ocr_and_merge(primary_path, secondary_path):
    """
    Performs OCR on two different image versions and merges the results for accuracy.
    - primary_path: The image enhanced specifically for OCR (grayscale, contrast).
    - secondary_path: The color image, corrected for perspective.
    """
    # For PDFs, we can only do a single pass
    if primary_path.lower().endswith('.pdf'):
        print(">>> OCR PASS: PDF detected, running single extraction.")
        return _call_gemini_api_single(primary_path)

    print(">>> OCR PASS 1 (Primary): Running on OCR-enhanced image.")
    primary_result = _call_gemini_api_single(primary_path)

    # Check if primary result is valid before proceeding
    if "error" in primary_result:
        print("!!! OCR PASS 1 FAILED. Returning error.")
        return primary_result

    print(">>> OCR PASS 2 (Secondary): Running on color-corrected image.")
    secondary_result = _call_gemini_api_single(secondary_path)

    # If the secondary pass fails, we can still return the primary result
    if "error" in secondary_result:
        print("!!! OCR PASS 2 FAILED. Falling back to primary result.")
        return primary_result

    # Both passes succeeded, merge the results
    return merge_ocr_results(primary_result, secondary_result)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def correct_perspective_and_enhance(image_path, output_folder):
    try:
        image = cv2.imread(image_path)
        if image is None: return image_path
        orig_height, orig_width = image.shape[:2]
        ratio = orig_height / 500.0
        image_resized = cv2.resize(image, (int(orig_width / ratio), 500))
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
        if screenCnt is None:
            print("!!! PERSPECTIVE CORRECTION: Could not find 4 corners. Using original.")
            return image_path
        pts = screenCnt.reshape(4, 2) * ratio
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        corrected_filename = f"corrected_{uuid.uuid4()}.jpg"
        corrected_filepath = os.path.join(output_folder, corrected_filename)
        cv2.imwrite(corrected_filepath, warped)
        print(f">>> PERSPECTIVE CORRECTION: Straightened image saved to {corrected_filepath}")
        return corrected_filepath
    except Exception as e:
        print(f"!!! PERSPECTIVE CORRECTION CRITICAL ERROR: {e}")
        return image_path


def preprocess_image_for_ocr(image_path, output_folder):
    try:
        img = cv2.imread(image_path)
        if img is None: return image_path
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, sharpening_kernel)
        processed_filename = f"processed_{uuid.uuid4()}.jpg"
        processed_filepath = os.path.join(output_folder, processed_filename)
        cv2.imwrite(processed_filepath, sharpened)
        print(f">>> IMAGE PREPROCESSING: Enhanced image saved to {processed_filepath}")
        return processed_filepath
    except Exception as e:
        print(f"!!! IMAGE PREPROCESSING CRITICAL ERROR: {e}")
        return image_path


def combine_images_to_pdf(image1_path, image2_path, output_folder):
    try:
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")
        pdf_filename = f"{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(output_folder, pdf_filename)
        img1.save(pdf_path, save_all=True, append_images=[img2])
        return pdf_path, pdf_filename
    except Exception as e:
        print(f"Error combining images to PDF: {e}")
        return None, None


def convert_pdf_first_page_to_image(pdf_path, output_folder):
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            doc.close();
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap()
        image_filename = f"pdf_page_{uuid.uuid4()}.jpg"
        image_filepath = os.path.join(output_folder, image_filename)
        pix.save(image_filepath)
        doc.close()
        return image_filepath
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        return None


def enhance_for_face_detection(image):
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpened_image


def detect_and_extract_face(image_path, output_folder):
    try:
        prototxt_path = 'deploy.prototxt'
        model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"!!! FACE DETECTION ERROR: Could not read image file at {image_path}")
            return None
        images_to_try = [("original", img), ("enhanced", enhance_for_face_detection(img))]
        for stage, current_img in images_to_try:
            print(f">>> FACE DETECTION (Pass: {stage}): Running detection...")
            (h, w) = current_img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(current_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            best_detection_index = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, best_detection_index, 2]
            if confidence > 0.2:
                print(f">>> FACE DETECTION SUCCESS (Stage: {stage}): Found face with confidence {confidence:.2f}")
                box = detections[0, 0, best_detection_index, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                padding = 20
                startX = max(0, startX - padding)
                startY = max(0, startY - padding)
                endX = min(w, endX + padding)
                endY = min(h, endY + padding)
                face_cropped = current_img[startY:endY, startX:endX]
                if face_cropped.size == 0: continue
                face_filename = f"face_{uuid.uuid4()}.jpg"
                face_filepath = os.path.join(output_folder, face_filename)
                cv2.imwrite(face_filepath, face_cropped)
                print(f">>> Face extracted to {face_filepath}")
                return face_filepath
            else:
                print(
                    f">>> FACE DETECTION (Pass: {stage}): No faces found with sufficient confidence (Max: {confidence:.2f}).")
        print("!!! FACE DETECTION FAILED: No face found after all enhancement passes.")
        return None
    except Exception as e:
        print(f"!!! FACE DETECTION CRITICAL ERROR: {e}")
        return None