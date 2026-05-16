import json
import re
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils.v6_order_utils import og_data_with_formats
from image_comparsion_siglip import compare_images


# =========================================================
# HELPERS
# =========================================================

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}


def normalize_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_for_compare(text):
    """Lowercase + collapse whitespace for fuzzy text comparison."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_urls(text):
    """Extract URLs, stripping any trailing punctuation."""
    if not text:
        return []
    urls = re.findall(r'https?://[^\s,;)\]\'"]+', text)
    return [u.rstrip('.,;:!?*') for u in urls]


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


def extract_phone_numbers(text):
    phones = []
    if not text:
        return phones
    phone_patterns = [
        r'\(\d{3}\)\s?\d{3}-\d{4}',
        r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',
        r'\+?\d{10,15}'
    ]
    for pattern in phone_patterns:
        phones.extend(re.findall(pattern, text))
    return phones


def extract_dates(text):
    if not text:
        return []
    date_patterns = [
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}'
    ]
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    return dates


def parse_date_to_tuple(date_str):
    """
    Normalize any date string into (year, month, day).
    Handles ISO, US slash, "Mon DD YYYY", and JS toString format
    (e.g. "Thu May 14 2026 14:00:02 GMT+0000...").
    """
    if not date_str:
        return None

    s = date_str.lower().strip()

    # YYYY-MM-DD
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # MM/DD/YYYY
    m = re.search(r'(\d{2})/(\d{2})/(\d{4})', s)
    if m:
        return (int(m.group(3)), int(m.group(1)), int(m.group(2)))

    # JS toString: "thu may 14 2026 ..."
    m = re.search(
        r'(?:mon|tue|wed|thu|fri|sat|sun)\s+'
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+'
        r'(\d{1,2})\s+(\d{4})',
        s
    )
    if m:
        return (int(m.group(3)), MONTH_MAP[m.group(1)], int(m.group(2)))

    # Mon[th] DD[,] YYYY
    m = re.search(
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})',
        s
    )
    if m:
        return (int(m.group(3)), MONTH_MAP[m.group(1)], int(m.group(2)))

    return None


def filter_empty_guidelines(guidelines):
    if not guidelines or not isinstance(guidelines, list):
        return []
    return [g for g in guidelines if g and isinstance(g, str) and g.strip()]


IMAGE_QUESTION_PATTERN = r'do you have an image[^?]*\?'
CAPTION_QUESTION_PATTERN = r'(?:do you want to change your caption|change your caption|new caption)'

# Customer answers that indicate "yes, change it" without providing actual
# replacement copy. These are treated as no edit so Post Content Validation
# doesn't false-fail demanding the literal word "change" appear in posts.
AMBIGUOUS_ANSWERS = {
    "", "change", "yes", "yes change", "yeah", "sure",
    "ok", "okay", "yes please", "please change", "change it"
}


QUESTION_START_RE = re.compile(
    r'^(do|does|did|are|is|was|were|can|could|will|would|should|'
    r'may|might|have|has|had|what|when|where|why|how|who)\b',
    re.IGNORECASE
)


def parse_qa_response(raw_response):
    """
    Parse a multi-question Atarim-style response into structured Q&A pairs.

    Format:
        * Question 1 *
         answer 1
        * Question 2 *
         answer 2

    Tolerates:
      - missing closing '*' on the final (or any) question — when this
        happens the question and its answer end up fused into one segment;
        we detect this by finding the '?' inside the segment
      - questions that end with '.' instead of '?'
      - any order of questions

    Returns:
        {
            "qa_pairs": [{"question": str, "answer": str}, ...],
            "caption_answer": str,         # answer to caption question (or "")
            "image_answer": str | None,    # raw answer text to image question;
                                           # None if image question not asked
            "image_url": str | None,       # URL extracted from image_answer
        }
    """
    if not raw_response:
        return {
            "qa_pairs": [],
            "caption_answer": "",
            "image_answer": None,
            "image_url": None,
        }

    segments = [s.strip() for s in raw_response.split('*')]
    if segments and not segments[0]:
        segments = segments[1:]

    qa_pairs = []
    i = 0
    while i < len(segments):
        seg = segments[i]

        if not seg:
            i += 1
            continue

        looks_like_question = QUESTION_START_RE.match(seg) is not None
        if not looks_like_question:
            i += 1
            continue

        # Case A: clean question — segment ends with '?' or '.' and the
        # answer lives in the next segment.
        if (seg.endswith('?') or seg.endswith('.')) and len(seg) < 200:
            answer = segments[i + 1].strip() if i + 1 < len(segments) else ""
            qa_pairs.append({"question": seg, "answer": answer})
            i += 2
            continue

        # Case B: fused — closing '*' was missing so question and answer
        # share one segment. Split at the first '?'.
        if '?' in seg:
            q_end = seg.index('?')
            question = seg[:q_end + 1].strip()
            answer = seg[q_end + 1:].strip()
            qa_pairs.append({"question": question, "answer": answer})
            i += 1
            continue

        i += 1

    # Classify pairs by question content
    caption_answer = ""
    image_answer = None
    image_url = None

    for pair in qa_pairs:
        q_lower = pair["question"].lower()

        if re.search(CAPTION_QUESTION_PATTERN, q_lower):
            caption_answer = pair["answer"]
        elif re.search(IMAGE_QUESTION_PATTERN, q_lower):
            image_answer = pair["answer"]
            url_match = re.search(
                r'https?://[^\s)\]\'"]+',
                pair["answer"]
            )
            if url_match:
                image_url = url_match.group(0).rstrip('.,;:!?*')

    return {
        "qa_pairs": qa_pairs,
        "caption_answer": caption_answer,
        "image_answer": image_answer,
        "image_url": image_url,
    }


def build_result(rule, passed, reason="", inputs=None, outputs=None):
    return {
        "rule": rule,
        "status": "PASS" if passed else "FAIL",
        "reason": reason,
        "input": inputs or {},
        "output": outputs or {}
    }


# =========================================================
# INTENT DETECTION
# =========================================================

def detect_intents(instruction):
    norm = normalize_text(instruction)

    intents = {
        "wants_website": (
            "website" in norm
            or "add url" in norm
            or "add link" in norm
        ),
        "remove_phone": (
            ("remove" in norm or "delete" in norm)
            and "phone" in norm
        ),
        "add_phone": (
            "add phone" in norm
            or "include phone" in norm
        ),
        "change_image": (
            "change image" in norm
            or "change photo" in norm
            or "replace image" in norm
            or "replace photo" in norm
            or "new image" in norm
            or "new photo" in norm
            or "update image" in norm
        ),
        "wants_customer_image": (
            ("customer" in norm and ("image" in norm or "photo" in norm))
            or "real photo" in norm
            or "actual photo" in norm
        ),
        "schedule_mentioned": (
            "schedule" in norm
            or "post on" in norm
            or "publish on" in norm
            or "go live" in norm
        ),
    }

    return intents


# =========================================================
# INDIVIDUAL CROSS-CHECKS
# =========================================================

def check_website_in_posts(final_posts):
    """
    Check website presence on posts that allow URLs.
    Instagram is exempt — IG posts must not contain URLs, so missing
    URL on IG is expected behavior, not a failure.
    """
    failures = []
    evidence = {}

    for platform, content in final_posts.items():
        urls = extract_urls(content)
        evidence[platform] = {"urls_found": urls}

        if platform == "instagram":
            evidence[platform]["note"] = (
                "instagram exempt — URLs not allowed; missing URL not flagged"
            )
            continue

        if len(urls) == 0:
            failures.append(f"{platform}: website requested but no URL in post")

    return failures, evidence


def check_phone_removed_in_posts(final_posts):
    failures = []
    evidence = {}

    for platform, content in final_posts.items():
        phones = extract_phone_numbers(content)
        evidence[platform] = {"phones_found": phones}
        if phones:
            failures.append(
                f"{platform}: phone removal requested but phone still present"
            )

    return failures, evidence


def check_phone_added_in_posts(final_posts):
    failures = []
    evidence = {}

    for platform, content in final_posts.items():
        phones = extract_phone_numbers(content)
        evidence[platform] = {"phones_found": phones}
        if not phones:
            failures.append(
                f"{platform}: phone addition requested but no phone in post"
            )

    return failures, evidence


def check_image_changed(image_url):
    failures = []
    evidence = {"image_url_present": bool(image_url), "image_url": image_url}

    if not image_url:
        failures.append("image change requested but image_url is empty")

    return failures, evidence


def check_image_against_guidelines(image_url, image_guidelines):
    failures = []

    requires_customer = False
    forbids_ai = False
    forbids_stock = False

    for g in image_guidelines:
        gl = g.lower()
        if "customer" in gl:
            requires_customer = True
        if "do not use ai" in gl or "no ai" in gl:
            forbids_ai = True
        if "no stock" in gl or "do not use stock" in gl:
            forbids_stock = True

    evidence = {
        "requires_customer": requires_customer,
        "forbids_ai": forbids_ai,
        "forbids_stock": forbids_stock,
        "image_url": image_url
    }

    if requires_customer and not image_url:
        failures.append(
            "customer image required by guidelines but image_url empty"
        )

    if image_url:
        url_lower = image_url.lower()
        if forbids_stock and (
            "shutterstock" in url_lower
            or "istock" in url_lower
            or "gettyimages" in url_lower
        ):
            failures.append(
                "guidelines forbid stock images but image_url appears to be stock"
            )
        if forbids_ai and (
            "ai-generated" in url_lower
            or "midjourney" in url_lower
            or "dalle" in url_lower
        ):
            failures.append(
                "guidelines forbid AI images but image_url appears AI-generated"
            )

    return failures, evidence


def check_schedule_against_instruction(instruction, schedule_date):
    failures = []

    instruction_dates = extract_dates(instruction)
    sched_tuple = parse_date_to_tuple(schedule_date)

    evidence = {
        "dates_in_instruction": instruction_dates,
        "schedule_date_raw": schedule_date,
        "schedule_date_parsed": sched_tuple,
        "comparisons": []
    }

    if instruction_dates and not schedule_date:
        failures.append("instruction mentions a date but schedule_date is empty")
        return failures, evidence

    for inst_date in instruction_dates:
        inst_tuple = parse_date_to_tuple(inst_date)
        comparison = {
            "instruction_date": inst_date,
            "instruction_parsed": inst_tuple,
            "schedule_parsed": sched_tuple,
            "match": None
        }

        if not inst_tuple or not sched_tuple:
            comparison["match"] = "skipped (unparseable)"
        elif inst_tuple != sched_tuple:
            comparison["match"] = False
            failures.append(
                f"instruction date '{inst_date}' does not match "
                f"schedule_date '{schedule_date}'"
            )
        else:
            comparison["match"] = True

        evidence["comparisons"].append(comparison)

    return failures, evidence


# =========================================================
# EDIT INSTRUCTION CONTENT COMPARISON
#
# Instagram rules:
#   - MUST NOT contain a URL (violation if present)
#   - Missing URL is allowed — not flagged
#   - Content is not otherwise compared (no residue match, no phrase
#     stripping)
#
# Facebook / Google:
#   - Must match target copy exactly
# =========================================================

def check_edit_instruction_content(edit_instruction, final_posts):
    """
    Compare each platform's post to the target copy.
    - facebook, google: must match target exactly (URLs included)
    - instagram: must NOT contain a URL; missing URLs are ignored
    """
    failures = []
    per_platform = {}

    if not edit_instruction:
        return failures, {"note": "no edit instruction copy to compare"}

    target_norm = normalize_for_compare(edit_instruction)
    target_urls = set(extract_urls(edit_instruction))

    for platform, content in final_posts.items():
        content_norm = normalize_for_compare(content)
        content_urls = set(extract_urls(content))

        platform_report = {
            "target_urls": list(target_urls),
            "post_urls": list(content_urls),
        }

        if platform == "instagram":
            if content_urls:
                platform_report["match"] = False
                platform_report["reason"] = (
                    f"instagram contains URL(s) which are not allowed: "
                    f"{', '.join(content_urls)}"
                )
                failures.append(
                    f"instagram: contains URL(s) not allowed on instagram "
                    f"({', '.join(content_urls)})"
                )
            else:
                platform_report["match"] = True
                platform_report["reason"] = (
                    "instagram has no URL (allowed); URL absence not flagged"
                )
        else:
            platform_report["exact_match"] = (content_norm == target_norm)
            platform_report["missing_urls"] = list(target_urls - content_urls)
            if not platform_report["exact_match"]:
                detail = (f" (missing URLs: {platform_report['missing_urls']})"
                          if platform_report["missing_urls"] else "")
                failures.append(
                    f"{platform}: does not match edit instruction copy{detail}"
                )

        per_platform[platform] = platform_report

    return failures, {"per_platform": per_platform}


# =========================================================
# VALIDATORS
# =========================================================

def validate_follow_instruction(
    follow_instruction,
    final_posts,
    schedule_date,
    image_url,
    image_guidelines
):
    """
    Runs intent detection on the follow_instruction and routes to
    the appropriate cross-checks.
    """
    inputs = {
        "follow_instruction": follow_instruction,
        "final_posts": final_posts,
        "schedule_date": schedule_date,
        "image_url": image_url,
        "image_guidelines": image_guidelines
    }

    if not follow_instruction:
        return build_result(
            "Follow Instruction Validation",
            True,
            "no follow instruction provided",
            inputs=inputs,
            outputs={"intents": {}, "routes_executed": [], "failures": []}
        )

    intents = detect_intents(follow_instruction)
    routes_executed = []
    all_failures = []
    route_evidence = {}

    if intents["wants_website"]:
        failures, evidence = check_website_in_posts(final_posts)
        all_failures += failures
        route_evidence["website_in_posts"] = evidence
        routes_executed.append("wants_website")

    if intents["remove_phone"] and not intents["add_phone"]:
        failures, evidence = check_phone_removed_in_posts(final_posts)
        all_failures += failures
        route_evidence["phone_removed"] = evidence
        routes_executed.append("remove_phone")

    if intents["add_phone"]:
        failures, evidence = check_phone_added_in_posts(final_posts)
        all_failures += failures
        route_evidence["phone_added"] = evidence
        routes_executed.append("add_phone")

    if intents["change_image"]:
        failures, evidence = check_image_changed(image_url)
        all_failures += failures
        route_evidence["image_changed"] = evidence
        routes_executed.append("change_image")

    if intents["change_image"] or intents["wants_customer_image"]:
        failures, evidence = check_image_against_guidelines(
            image_url, image_guidelines
        )
        all_failures += failures
        route_evidence["image_guidelines"] = evidence
        routes_executed.append("image_guidelines")

    if intents["schedule_mentioned"]:
        failures, evidence = check_schedule_against_instruction(
            follow_instruction, schedule_date
        )
        all_failures += failures
        route_evidence["schedule"] = evidence
        routes_executed.append("schedule")

    outputs = {
        "intents": intents,
        "routes_executed": routes_executed,
        "evidence": route_evidence,
        "failures": all_failures
    }

    return build_result(
        "Follow Instruction Validation",
        len(all_failures) == 0,
        "; ".join(all_failures),
        inputs=inputs,
        outputs=outputs
    )


def validate_schedule(schedule_date, final_posts):
    inputs = {"schedule_date": schedule_date}

    if not schedule_date:
        return build_result(
            "Schedule Validation",
            False,
            "schedule date missing",
            inputs=inputs,
            outputs={"schedule_parsed": None, "dates_in_posts": []}
        )

    sched_tuple = parse_date_to_tuple(schedule_date)
    combined_text = " ".join(final_posts.values())
    dates_in_posts = extract_dates(combined_text)

    mismatches = []
    comparisons = []

    for detected in dates_in_posts:
        detected_tuple = parse_date_to_tuple(detected)
        match = None
        if detected_tuple and sched_tuple:
            match = detected_tuple == sched_tuple
            if not match:
                mismatches.append(detected)
        comparisons.append({
            "date_in_post": detected,
            "post_parsed": detected_tuple,
            "schedule_parsed": sched_tuple,
            "match": match
        })

    outputs = {
        "schedule_parsed": sched_tuple,
        "dates_in_posts": dates_in_posts,
        "comparisons": comparisons,
        "mismatches": mismatches
    }

    if mismatches:
        return build_result(
            "Schedule Validation",
            False,
            f"date mismatch in post content: {', '.join(mismatches)}",
            inputs=inputs,
            outputs=outputs
        )

    return build_result(
        "Schedule Validation",
        True,
        inputs=inputs,
        outputs=outputs
    )


def validate_jira_schedule_match(schedule_date, jira_descriptions):
    """
    Validates that the schedule_date from complete_toggle_data matches
    the date in jira_descriptions field_value.
    """
    inputs = {
        "schedule_date": schedule_date,
        "jira_descriptions": jira_descriptions
    }

    if not schedule_date:
        return build_result(
            "Jira Schedule Match Validation",
            False,
            "schedule_date missing from complete_toggle_data",
            inputs=inputs,
            outputs={"jira_date_parsed": None, "schedule_parsed": None}
        )

    if not jira_descriptions or len(jira_descriptions) == 0:
        return build_result(
            "Jira Schedule Match Validation",
            False,
            "jira_descriptions missing or empty",
            inputs=inputs,
            outputs={"jira_date_parsed": None, "schedule_parsed": None}
        )

    # Extract date from jira_descriptions field_value
    field_value = jira_descriptions[0].get("field_value", "")
    if not field_value:
        return build_result(
            "Jira Schedule Match Validation",
            False,
            "field_value missing in jira_descriptions",
            inputs=inputs,
            outputs={"jira_date_parsed": None, "schedule_parsed": None}
        )

    # Extract only the date after "Scheduled Date:" pattern
    # Pattern matches: "Scheduled Date:\nYYYY-MM-DD HH:MM:SS TZ"
    jira_date_match = re.search(
        r'Scheduled Date:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})',
        field_value,
        re.IGNORECASE
    )

    if not jira_date_match:
        return build_result(
            "Jira Schedule Match Validation",
            False,
            "no 'Scheduled Date:' found in jira_descriptions field_value",
            inputs=inputs,
            outputs={
                "jira_field_value": field_value,
                "jira_date_extracted": None,
                "schedule_parsed": parse_date_to_tuple(schedule_date)
            }
        )

    jira_date_str = jira_date_match.group(1)

    # Parse both dates
    sched_tuple = parse_date_to_tuple(schedule_date)
    jira_tuple = parse_date_to_tuple(jira_date_str)

    outputs = {
        "schedule_date_raw": schedule_date,
        "schedule_parsed": sched_tuple,
        "jira_field_value": field_value,
        "jira_date_extracted": jira_date_str,
        "jira_date_parsed": jira_tuple,
        "match": None
    }

    if not sched_tuple or not jira_tuple:
        outputs["match"] = "skipped (unparseable)"
        return build_result(
            "Jira Schedule Match Validation",
            False,
            "unable to parse one or both dates",
            inputs=inputs,
            outputs=outputs
        )

    outputs["match"] = sched_tuple == jira_tuple

    if not outputs["match"]:
        return build_result(
            "Jira Schedule Match Validation",
            False,
            f"schedule_date '{schedule_date}' does not match jira_descriptions date '{jira_date_str}'",
            inputs=inputs,
            outputs=outputs
        )

    return build_result(
        "Jira Schedule Match Validation",
        True,
        inputs=inputs,
        outputs=outputs
    )


def validate_post_content(final_posts, edit_instruction="", caption_answer=""):
    """
    Post Content Validation
      - Side 1: each platform vs. edit_instruction (target copy)
      - Side 2: each platform vs. Facebook (cross-platform consistency)

    Instagram is exempt from text comparison on both sides.
    Only the no-URL rule applies to Instagram.

    Parameters:
        edit_instruction: the copy actually compared against posts. Empty
            when caption_answer is ambiguous ("change", "yes", etc.).
        caption_answer: raw customer answer from the Q&A response, surfaced
            for audit/debugging even when not used in comparison.
    """
    inputs = {
        "final_posts": final_posts,
        "edit_instruction": edit_instruction,
        "caption_answer": caption_answer
    }

    empties = [
        platform for platform, content in final_posts.items()
        if not content or not content.strip()
    ]

    if empties:
        return build_result(
            "Post Content Validation",
            False,
            f"empty content: {', '.join(empties)}",
            inputs=inputs,
            outputs={"empty_platforms": empties}
        )

    # --- Side 1: each platform vs. edit_instruction (target copy) ---
    edit_failures, edit_evidence = check_edit_instruction_content(
        edit_instruction, final_posts
    )
    edit_per_platform = edit_evidence.get("per_platform", {}) if isinstance(
        edit_evidence, dict
    ) else {}

    # --- Side 2: each platform vs. Facebook ---
    reference = final_posts.get("facebook", "")
    fb_failures = []
    fb_per_platform = {}

    for platform, content in final_posts.items():
        if platform == "facebook":
            fb_per_platform[platform] = {"role": "reference"}
            continue

        if platform == "instagram":
            ig_urls = extract_urls(content)
            fb_per_platform[platform] = {
                "skipped": True,
                "reason": (
                    "instagram content not compared to facebook; "
                    "only no-URL rule enforced"
                ),
                "urls_found": ig_urls,
                "url_count": len(ig_urls),
                "compliant": len(ig_urls) == 0
            }
            if ig_urls:
                fb_failures.append(
                    f"instagram contains URL(s) not allowed on instagram: "
                    f"{', '.join(ig_urls)}"
                )
            continue

        # google and any other platform: must match facebook exactly
        exact_match = content == reference
        fb_per_platform[platform] = {
            "exact_match": exact_match,
            "length_post": len(content),
            "length_facebook": len(reference)
        }
        if not exact_match:
            fb_failures.append(f"{platform}: content differs from Facebook")

    # --- Merge per-platform view ---
    comparisons = {}
    for platform in final_posts.keys():
        comparisons[platform] = {
            "vs_edit_instruction": edit_per_platform.get(platform, {}),
            "vs_facebook": fb_per_platform.get(platform, {})
        }

    all_failures = edit_failures + fb_failures

    # Audit note: explain when raw answer wasn't used for comparison
    caption_note = None
    if caption_answer and not edit_instruction:
        caption_note = (
            f"caption_answer {caption_answer!r} treated as ambiguous "
            f"(no replacement copy); edit_instruction comparison skipped"
        )
    elif caption_answer and edit_instruction:
        caption_note = "caption_answer used as edit_instruction for comparison"
    elif not caption_answer:
        caption_note = "no caption answer provided in response"

    return build_result(
        "Post Content Validation",
        len(all_failures) == 0,
        "; ".join(all_failures),
        inputs=inputs,
        outputs={
            "caption_answer_note": caption_note,
            "comparisons": comparisons,
            "inconsistencies": all_failures
        }
    )


def validate_image_guidelines(
    image_guidelines,
    image_url,
    requested_image_url=None,
    image_answer=None
):
    """
    Two independent sub-checks, both contributing to PASS/FAIL:
      1. Guideline rules (only evaluated if guidelines exist)
      2. Image comparison: imageUrlP1 vs requested image from edit
         instructions (runs whenever both URLs are available, regardless
         of whether guidelines are defined)

    Parameters:
        image_answer: raw text answer to "Do you have an image..." question.
            - None  → question was not asked in the response
            - ""    → question asked but no answer given
            - other → answer text (URL, "no thanks", etc.)
    """
    inputs = {
        "image_guidelines": image_guidelines,
        "image_url": image_url,
        "requested_image_url": requested_image_url,
        "image_answer": image_answer
    }

    failures = []
    outputs = {}

    # --- Sub-check 1: guideline rules ---
    if image_guidelines:
        g_failures, g_evidence = check_image_against_guidelines(
            image_url, image_guidelines
        )
        failures += g_failures
        outputs["guideline_evidence"] = g_evidence
        outputs["guideline_failures"] = g_failures
    else:
        outputs["guideline_evidence"] = {"note": "no guidelines defined"}
        outputs["guideline_failures"] = []

    # --- Sub-check 2: image comparison (independent of guidelines) ---
    comparison_result = None
    image_instruction_note = None

    if image_url and requested_image_url:
        print(
            f"[IMAGE COMP] Comparing imageUrlP1={image_url} "
            f"with requested image={requested_image_url}"
        )
        try:
            comparison_result = compare_images(image_url, requested_image_url)
            print(f"[IMAGE COMP] Comparison result: {comparison_result}")
        except Exception as e:
            print(f"[IMAGE COMP] Comparison failed: {e}")
            comparison_result = {"error": str(e), "result": "ERROR"}

        if comparison_result:
            result_label = str(comparison_result.get("result", "")).upper()
            if result_label in ("DIFFERENT", "MISMATCH", "NO_MATCH"):
                failures.append(
                    "imageUrlP1 does not match the requested image from "
                    "edit instructions"
                )
            elif result_label == "ERROR":
                failures.append(
                    f"image comparison failed: "
                    f"{comparison_result.get('error', 'unknown error')}"
                )
    elif image_answer is not None:
        # Image question was asked but we couldn't run comparison
        if not requested_image_url:
            image_instruction_note = (
                "image question present but no URL provided in response"
            )
        elif not image_url:
            image_instruction_note = (
                "requested image present but imageUrlP1 is empty"
            )
    else:
        image_instruction_note = "no image instruction in response"

    outputs["image_comparison"] = comparison_result
    outputs["image_instruction_note"] = image_instruction_note

    if failures:
        reason = "; ".join(failures)
    elif image_guidelines or comparison_result:
        reason = "image checks passed"
    else:
        reason = "no image rules or comparison applicable"

    return build_result(
        "Image Validation",
        len(failures) == 0,
        reason,
        inputs=inputs,
        outputs=outputs
    )


# =========================================================
# ORDER TYPE NORMALIZATION
# =========================================================

def validate_order_type(iris_id):
    patterns = [
        r'^SOCIAL-\d+$',
        r'^Training 1\.0 - SOCIAL-\d+$',
        r'^Pre-live SOCIAL-\d+$'
    ]
    return any(re.match(p, iris_id) for p in patterns)


def extract_base_iris_id(iris_id):
    match = re.search(r'SOCIAL-\d+', iris_id)
    return match.group(0) if match else iris_id


# =========================================================
# MAIN REPORT
# =========================================================

def generate_validation_report(iris_id):
    try:
        order_result = og_data_with_formats(iris_id)
        data = order_result['data']
    except Exception as e:
        return {
            "order_id": iris_id,
            "overall_status": "ERROR",
            "inputs": {},
            "validations": [],
            "error": str(e)
        }

    v6 = data.get("v6_response", {})

    # POST CONTENT
    post_content = v6.get("post_content", {})
    final_posts = {
        "facebook": post_content.get("fb_postcontent1", ""),
        "instagram": post_content.get("insta_postcontent1", ""),
        "google": post_content.get("google_postcontent1", "")
    }

    # EDIT INSTRUCTIONS
    raw_response = ""
    edit_instructions = v6.get("edit_instructions", [])
    if edit_instructions:
        raw_response = edit_instructions[0].get("response", "")

    follow_instruction, edit_instruction, requested_image_url, qa_pairs, \
        caption_answer = "", "", None, [], ""

    qa = parse_qa_response(raw_response)
    qa_pairs = qa["qa_pairs"]
    caption_answer = qa["caption_answer"]
    image_answer = qa["image_answer"]
    requested_image_url = qa["image_url"]

    # follow_instruction = the first question itself (kept for
    # backward-compat with detect_intents which reads keywords)
    follow_instruction = qa_pairs[0]["question"] if qa_pairs else ""

    # edit_instruction = ONLY the caption answer. Image questions and their
    # URLs are routed to Image Validation, not Post Content Validation.
    # If the customer answered with vague text like "change" or "yes" (no
    # actual replacement copy), treat as no edit so Post Content Validation
    # doesn't false-fail.
    if caption_answer.lower().strip() in AMBIGUOUS_ANSWERS:
        edit_instruction = ""
    else:
        edit_instruction = caption_answer

    # COMPLETE TOGGLE DATA
    complete_data = v6.get("complete_toggle_data") or {}
    schedule_date = complete_data.get("scheduleDate1", "")
    image_url = complete_data.get("imageUrlP1", "")

    # JIRA DESCRIPTIONS
    jira_descriptions = v6.get("jira_descriptions", [])

    # BRAND GUIDE
    image_guidelines = []
    brand_guide = v6.get("brand_guide", [])
    if brand_guide:
        try:
            bg_data = json.loads(brand_guide[0].get("data", "{}"))
            image_guidelines = bg_data.get("imageGuidelines", [])
        except Exception as e:
            print("Brand guide parsing failed:", e)
    image_guidelines = filter_empty_guidelines(image_guidelines)

    # INPUTS SNAPSHOT
    inputs_snapshot = {
        "raw_edit_response": raw_response,
        "qa_pairs": qa_pairs,
        "caption_answer": caption_answer,
        "image_answer": image_answer,
        "follow_instruction": follow_instruction,
        "edit_instruction": edit_instruction,
        "requested_image_url": requested_image_url,
        "final_posts": final_posts,
        "schedule_date": schedule_date,
        "image_url": image_url,
        "image_guidelines": image_guidelines,
        "jira_descriptions": jira_descriptions
    }

    # RUN VALIDATIONS
    validations = [
        validate_follow_instruction(
            follow_instruction=follow_instruction,
            final_posts=final_posts,
            schedule_date=schedule_date,
            image_url=image_url,
            image_guidelines=image_guidelines
        ),
        validate_schedule(schedule_date, final_posts),
        validate_jira_schedule_match(schedule_date, jira_descriptions),
        validate_post_content(
            final_posts,
            edit_instruction=edit_instruction,
            caption_answer=caption_answer
        ),
        validate_image_guidelines(
            image_guidelines=image_guidelines,
            image_url=image_url,
            requested_image_url=requested_image_url,
            image_answer=image_answer
        )
    ]

    overall_pass = all(v["status"] == "PASS" for v in validations)

    return {
        "order_id": data.get("iris_id"),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "inputs": inputs_snapshot,
        "validations": validations
    }


# =========================================================
# FASTAPI
# =========================================================

class ValidationRequest(BaseModel):
    iris_id: str


app = FastAPI(
    title="Social Media Validation API",
    description="Validates social media post content against business rules and guidelines"
)


@app.post("/validate")
def validate_order(request: ValidationRequest):
    if not validate_order_type(request.iris_id):
        return JSONResponse(
            status_code=400,
            content={
                "error": "unsupported order id format",
                "iris_id": request.iris_id
            }
        )

    base_id = extract_base_iris_id(request.iris_id)
    result = generate_validation_report(base_id)

    if "error" in result:
        return JSONResponse(status_code=404, content=result)

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)