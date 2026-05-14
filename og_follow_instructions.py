import json
import re
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from utils.v6_order_utils import og_data_with_formats


# =========================================================
# HELPERS
# =========================================================

def normalize_text(text):
    if not text:
        return ""

    return re.sub(r"\s+", " ", text.strip().lower())


def extract_urls(text):

    if not text:
        return []

    return re.findall(r'https?://\S+', text)


def contains_phone_number(text):

    phone_patterns = [
        r'\(\d{3}\)\s?\d{3}-\d{4}',
        r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',
        r'\+?\d{10,15}'
    ]

    for pattern in phone_patterns:

        if re.search(pattern, text):
            return True

    return False


def extract_dates(text):

    date_patterns = [
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}'
    ]

    dates = []

    for pattern in date_patterns:

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        dates.extend(matches)

    return dates


# =========================================================
# STATUS HELPERS
# =========================================================

def build_result(
    rule,
    passed,
    reason=""
):

    return {
        "rule": rule,
        "status": "PASS" if passed else "FAIL",
        "reason": reason
    }


# =========================================================
# VALIDATION FUNCTIONS
# =========================================================

def validate_edit_instruction(
    instruction,
    final_posts
):

    print(f"\n=== EDIT INSTRUCTION VALIDATION ===")
    print(f"Instruction: {instruction}")
    
    instruction = normalize_text(instruction)
    print(f"Normalized: {instruction}")

    wants_website = (
        "website" in instruction
    )

    remove_phone = (
        "phone number" in instruction
        or "remove phone" in instruction
    )

    change_image = (
        "change image" in instruction
        or "change photo" in instruction
        or "replace image" in instruction
    )

    print(f"Wants website: {wants_website}")
    print(f"Remove phone: {remove_phone}")
    print(f"Change image: {change_image}")

    all_pass = True
    reasons = []

    for platform, content in final_posts.items():
        print(f"\n--- Checking {platform} ---")
        print(f"Content: {content[:100]}...")

        urls = extract_urls(content)
        has_phone = contains_phone_number(content)

        print(f"URLs found: {len(urls)}")
        print(f"Phone detected: {has_phone}")

        # =====================================================
        # WEBSITE CHECK
        # =====================================================

        if wants_website:

            if len(urls) == 0:

                all_pass = False

                reasons.append(
                    f"{platform}: website missing"
                )
                print(f"❌ Website missing for {platform}")

        # =====================================================
        # PHONE REMOVAL CHECK
        # =====================================================

        if remove_phone:

            if has_phone:

                all_pass = False

                reasons.append(
                    f"{platform}: phone still present"
                )
                print(f"❌ Phone still present for {platform}")

    # =========================================================
    # IMAGE CHANGE CHECK
    # =========================================================

    if change_image:

        reasons.append(
            "cannot verify image change automatically"
        )
        print(f"⚠️ Image change requested - cannot verify automatically")

    result = build_result(
        "Edit Instruction Validation",
        all_pass,
        ", ".join(reasons)
    )
    
    print(f"Result: {result['status']} - {result['reason']}")
    print(f"=== END EDIT INSTRUCTION VALIDATION ===\n")
    
    return result


# =========================================================

def validate_schedule(
    schedule_date,
    final_posts
):

    print(f"\n=== SCHEDULE VALIDATION ===")
    print(f"Schedule date: {schedule_date}")
    
    if not schedule_date:

        result = build_result(
            "Schedule Validation",
            False,
            "schedule date missing"
        )
        print(f"❌ Schedule date missing")
        print(f"Result: {result['status']} - {result['reason']}")
        print(f"=== END SCHEDULE VALIDATION ===\n")
        return result

    combined_text = " ".join(
        final_posts.values()
    )
    print(f"Combined text: {combined_text[:200]}...")

    dates = extract_dates(combined_text)
    print(f"Dates found in content: {dates}")

    for detected in dates:

        if detected.lower() not in schedule_date.lower():
            result = build_result(
                "Schedule Validation",
                False,
                f"date mismatch detected: {detected}"
            )
            print(f"❌ Date mismatch: {detected}")
            print(f"Result: {result['status']} - {result['reason']}")
            print(f"=== END SCHEDULE VALIDATION ===\n")
            return result

    result = build_result(
        "Schedule Validation",
        True
    )
    print(f"✅ Schedule validation passed")
    print(f"Result: {result['status']} - {result['reason']}")
    print(f"=== END SCHEDULE VALIDATION ===\n")
    return result


# =========================================================

def validate_implementation_guidelines(
    guidelines,
    final_posts
):

    print(f"\n=== IMPLEMENTATION GUIDELINE VALIDATION ===")
    print(f"Guidelines: {guidelines}")
    
    combined_text = " ".join(
        final_posts.values()
    ).lower()
    print(f"Combined text: {combined_text[:200]}...")

    violations = []

    for guideline in guidelines:

        guideline = guideline.lower()
        print(f"\n--- Checking guideline: {guideline} ---")

        # =====================================================
        # RESTRICTED WORDS
        # =====================================================

        restricted_words = [
            "holistic",
            "expert"
        ]

        if (
            "do not use the word"
            in guideline
        ):

            for word in restricted_words:

                if word in combined_text:

                    violations.append(
                        f"restricted word found: {word}"
                    )
                    print(f"❌ Restricted word found: {word}")

        # =====================================================
        # NO SKIN CARE
        # =====================================================

        if (
            "skin care"
            in guideline
        ):

            if (
                "skin care"
                in combined_text
            ):

                violations.append(
                    "skin care mentioned"
                )
                print(f"❌ Skin care mentioned")

        # =====================================================
        # NO PRICING
        # =====================================================

        if (
            "do not focus on pricing"
            in guideline
        ):

            pricing_words = [
                "%",
                "discount",
                "sale",
                "$",
                "offer"
            ]

            for word in pricing_words:

                if word in combined_text:

                    violations.append(
                        f"pricing detected: {word}"
                    )
                    print(f"❌ Pricing word detected: {word}")

    result = build_result(
        "Implementation Guideline Validation",
        len(violations) == 0,
        ", ".join(violations)
    )
    
    print(f"Violations: {violations}")
    print(f"Result: {result['status']} - {result['reason']}")
    print(f"=== END IMPLEMENTATION GUIDELINE VALIDATION ===\n")
    
    return result


# =========================================================

def validate_post_content(
    final_posts
):

    print(f"\n=== POST CONTENT VALIDATION ===")
    
    # Check for empty content first
    for platform, content in final_posts.items():
        print(f"--- Checking {platform} ---")
        print(f"Content length: {len(content)}")
        print(f"Content: {content[:50]}...")

        if not content:
            result = build_result(
                "Post Content Validation",
                False,
                f"{platform}: empty content"
            )
            print(f"❌ Empty content for {platform}")
            print(f"Result: {result['status']} - {result['reason']}")
            print(f"=== END POST CONTENT VALIDATION ===\n")
            return result

    print(f"✅ All platforms have content")
    
    # Now check content consistency
    print(f"\n--- Checking Content Consistency ---")
    
    # Get reference content (Facebook)
    reference_content = final_posts.get("facebook", "")
    print(f"Reference (Facebook): {reference_content[:100]}...")
    
    inconsistencies = []
    
    for platform, content in final_posts.items():
        if platform == "facebook":
            continue  # Skip reference
            
        print(f"--- Comparing {platform} ---")
        print(f"Content: {content[:100]}...")
        
        if content != reference_content:
            inconsistencies.append(f"{platform}: content differs from Facebook")
            print(f"❌ {platform} content differs from Facebook")
        else:
            print(f"✅ {platform} content matches Facebook")
    
    # Build combined result
    all_reasons = []
    
    if inconsistencies:
        all_reasons.extend(inconsistencies)
    
    result = build_result(
        "Post Content Validation",
        len(inconsistencies) == 0,
        ", ".join(all_reasons)
    )
    
    print(f"Inconsistencies: {inconsistencies}")
    print(f"Result: {result['status']} - {result['reason']}")
    print(f"=== END POST CONTENT VALIDATION ===\n")
    
    return result


# =========================================================

def validate_content_consistency(
    final_posts
):

    print(f"\n=== CONTENT CONSISTENCY VALIDATION ===")
    
    # Get reference content (Facebook)
    reference_content = final_posts.get("facebook", "")
    print(f"Reference (Facebook): {reference_content[:100]}...")
    
    inconsistencies = []
    
    for platform, content in final_posts.items():
        if platform == "facebook":
            continue  # Skip reference
            
        print(f"--- Checking {platform} ---")
        print(f"Content: {content[:100]}...")
        
        if content != reference_content:
            inconsistencies.append(f"{platform}: content differs from Facebook")
            print(f"❌ {platform} content differs from Facebook")
        else:
            print(f"✅ {platform} content matches Facebook")
    
    result = build_result(
        "Content Consistency Validation",
        len(inconsistencies) == 0,
        ", ".join(inconsistencies)
    )
    
    print(f"Inconsistencies: {inconsistencies}")
    print(f"Result: {result['status']} - {result['reason']}")
    print(f"=== END CONTENT CONSISTENCY VALIDATION ===\n")
    
    return result


# =========================================================

def validate_image_guidelines(
    image_guidelines,
    image_url
):

    print(f"\n=== IMAGE VALIDATION ===")
    print(f"Image guidelines: {image_guidelines}")
    print(f"Image URL: {image_url}")

    requires_customer_image = False

    for guideline in image_guidelines:

        guideline = guideline.lower()
        print(f"--- Checking guideline: {guideline} ---")

        if (
            "customer"
            in guideline
            or "do not use ai"
            in guideline
            or "stock photo"
            in guideline
        ):

            requires_customer_image = True
            print(f"✅ Customer image required")

    # =========================================================
    # IMAGE REQUIRED
    # =========================================================

    if requires_customer_image:

        if not image_url:

            result = build_result(
                "Image Validation",
                False,
                "image missing"
            )
            print(f"❌ Image missing but required")
            print(f"Result: {result['status']} - {result['reason']}")
            print(f"=== END IMAGE VALIDATION ===\n")
            return result

    result = build_result(
        "Image Validation",
        True
    )
    print(f"✅ Image validation passed")
    print(f"Result: {result['status']} - {result['reason']}")
    print(f"=== END IMAGE VALIDATION ===\n")
    return result


# =========================================================
# MAIN REPORT FUNCTION
# =========================================================

def generate_validation_report(
    iris_id
):

    # Use the utility function to fetch order details with multiple format support
    try:
        order_result = og_data_with_formats(iris_id)
        data = order_result['data']
    except Exception as e:
        return {
            "order_id": iris_id,
            "overall_status": "ERROR",
            "validations": [],
            "error": str(e)
        }

    v6 = data.get(
        "v6_response",
        {}
    )

    # =====================================================
    # POST CONTENT
    # =====================================================

    post_content = v6.get(
        "post_content",
        {}
    )

    final_posts = {
        "facebook": post_content.get(
            "fb_postcontent1",
            ""
        ),

        "instagram": post_content.get(
            "insta_postcontent1",
            ""
        ),

        "google": post_content.get(
            "google_postcontent1",
            ""
        )
    }

    # =====================================================
    # EDIT INSTRUCTIONS
    # =====================================================

    edit_instruction = ""

    edit_instructions = v6.get(
        "edit_instructions",
        []
    )

    if edit_instructions:

        edit_instruction = (
            edit_instructions[0].get(
                "response",
                ""
            )
        )

    # =====================================================
    # COMPLETE DATA
    # =====================================================

    complete_data = v6.get(
        "complete_toggle_data"
    )

    schedule_date = ""

    image_url = ""

    if complete_data:

        schedule_date = (
            complete_data.get(
                "scheduleDate1",
                ""
            )
        )

        image_url = (
            complete_data.get(
                "imageUrlP1",
                ""
            )
        )

    # =====================================================
    # BRAND GUIDE
    # =====================================================

    implementation_guidelines = []

    image_guidelines = []

    brand_guide = v6.get(
        "brand_guide",
        []
    )

    if brand_guide:

        try:

            bg_data = json.loads(
                brand_guide[0].get(
                    "data",
                    "{}"
                )
            )

            implementation_guidelines = (
                bg_data.get(
                    "implementationGuidelines",
                    []
                )
            )

            image_guidelines = (
                bg_data.get(
                    "imageGuidelines",
                    []
                )
            )

        except Exception as e:

            print(
                "Brand guide parsing failed:",
                e
            )

    # =====================================================
    # RUN VALIDATIONS
    # =====================================================

    validations = []

    validations.append(
        validate_edit_instruction(
            edit_instruction,
            final_posts
        )
    )

    validations.append(
        validate_schedule(
            schedule_date,
            final_posts
        )
    )

    validations.append(
        validate_post_content(
            final_posts
        )
    )

    validations.append(
        validate_image_guidelines(
            image_guidelines,
            image_url
        )
    )

    # =====================================================
    # OVERALL STATUS
    # =====================================================

    overall_pass = all(
        v["status"] == "PASS"
        for v in validations
    )

    overall_status = (
        "PASS"
        if overall_pass
        else "FAIL"
    )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    report = {
        "order_id": data.get(
            "iris_id"
        ),

        "overall_status": overall_status,

        "validations": validations
    }

    return report


# =========================================================
# ORDER TYPE VALIDATION
# =========================================================

def validate_order_type(iris_id: str) -> bool:
    """
    Validate if the iris_id matches supported order types:
    - Standard format: SOCIAL-XXXXXXX
    - Training format: Training 1.0 - SOCIAL-XXXXXXX
    - Pre-live format: Pre-live SOCIAL-XXXXXXX
    """
    # Standard SOCIAL format
    standard_pattern = r'^SOCIAL-\d+$'
    
    # Training 1.0 format
    training_pattern = r'^Training 1\.0 - SOCIAL-\d+$'
    
    # Pre-live format
    prelive_pattern = r'^Pre-live SOCIAL-\d+$'
    
    return (re.match(standard_pattern, iris_id) or 
            re.match(training_pattern, iris_id) or 
            re.match(prelive_pattern, iris_id))

def extract_base_iris_id(iris_id: str) -> str:
    """
    Extract the base SOCIAL ID from various order types:
    - Standard: SOCIAL-879245 -> SOCIAL-879245
    - Training: Training 1.0 - SOCIAL-812912 -> SOCIAL-812912
    - Pre-live: Pre-live SOCIAL-920598 -> SOCIAL-920598
    """
    # Extract base SOCIAL ID from any format
    match = re.search(r'SOCIAL-\d+', iris_id)
    return match.group(0) if match else iris_id

# =========================================================
# REQUEST MODEL
# =========================================================

from pydantic import BaseModel

class ValidationRequest(BaseModel):
    iris_id: str

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Social Media Validation API",
    description="Validates social media post content against business rules and guidelines"
)


@app.post("/validate")
def validate_order(request: ValidationRequest):
    """
    Validate social media content for a given order ID
    
    Supports order types:
    - Standard: SOCIAL-XXXXXXX
    - Training 1.0: Training 1.0 - SOCIAL-XXXXXXX
    - Pre-live: Pre-live SOCIAL-XXXXXXX
    
    Args:
        iris_id: The order ID to validate (e.g., "879761")
    
    Returns:
        JSON validation report with overall status and detailed validations
    """
    result = generate_validation_report(request.iris_id)
    
    if "error" in result:
        return JSONResponse(
            status_code=404,
            content=result
        )
    
    return result

# =========================================================
# MAIN (for local testing)
# =========================================================

if __name__ == "__main__":

    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)