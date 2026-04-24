import csv
import html
import json
import re
import urllib.request
from pathlib import Path

PLOTS_DIR = Path("plots")
INPUT_CSV = PLOTS_DIR / "feature_metadata.csv"
OUTPUT_CSV = PLOTS_DIR / "feature_metadata_enriched.csv"
OUTPUT_JSON = PLOTS_DIR / "feature_metadata_enriched.json"
MODEL_ID = "gemma-2-2b"
SOURCE_ID = "12-gemmascope-res-16k"
FEATURE_API_BASE = "https://www.neuronpedia.org/api/feature"
FEATURE_PAGE_BASE = "https://www.neuronpedia.org"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def extract_positive_logits_hint(feature_obj: dict, max_items: int = 5) -> str:
    pos_tokens = feature_obj.get("pos_str", [])
    if not isinstance(pos_tokens, list):
        return ""
    cleaned = []
    for tok in pos_tokens:
        if not isinstance(tok, str):
            continue
        t = clean_text(tok)
        if t:
            cleaned.append(t)
        if len(cleaned) >= max_items:
            break
    return ", ".join(cleaned)


def extract_explanation(feature_obj: dict) -> str:
    explanations = feature_obj.get("explanations", [])
    if isinstance(explanations, list):
        for exp in explanations:
            if not isinstance(exp, dict):
                continue
            desc = clean_text(str(exp.get("description", "")))
            if desc:
                return desc
    return "unavailable"


def fetch_feature_metadata(feature_id: int) -> dict:
    api_url = f"{FEATURE_API_BASE}/{MODEL_ID}/{SOURCE_ID}/{feature_id}"
    page_url = f"{FEATURE_PAGE_BASE}/{MODEL_ID}/{SOURCE_ID}/{feature_id}"
    result = {
        "feature_id": feature_id,
        "url": page_url,
        "api_url": api_url,
        "status": "ok",
        "explanation": "unavailable",
        "positive_logits_hint": "",
        "title": "",
        "error": "",
    }
    try:
        req = urllib.request.Request(
            api_url,
            headers={"Accept": "application/json", "User-Agent": "python-urllib"},
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            feature_obj = json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    result["title"] = f"{feature_obj.get('modelId', MODEL_ID)} · {feature_obj.get('layer', SOURCE_ID)} · {feature_obj.get('index', feature_id)}"
    result["explanation"] = extract_explanation(feature_obj)
    hint = extract_positive_logits_hint(feature_obj)
    if hint:
        result["positive_logits_hint"] = hint

    if result["explanation"] == "unavailable" and hint:
        result["explanation"] = f"logit tokens: {hint}"

    return result


def read_feature_ids(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ids = []
        for row in reader:
            value = row.get("feature_id", "").strip()
            if value.isdigit():
                ids.append(int(value))
        return ids


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    feature_ids = read_feature_ids(INPUT_CSV)
    if not feature_ids:
        raise ValueError(f"No feature_id values found in {INPUT_CSV}")

    # Keep input order while dropping duplicates.
    seen = set()
    ordered_ids = []
    for fid in feature_ids:
        if fid not in seen:
            seen.add(fid)
            ordered_ids.append(fid)

    rows = []
    for i, fid in enumerate(ordered_ids, start=1):
        print(f"[{i}/{len(ordered_ids)}] fetching f{fid}")
        rows.append(fetch_feature_metadata(fid))

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_id",
                "status",
                "explanation",
                "positive_logits_hint",
                "title",
                "url",
                "api_url",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Saved: {OUTPUT_CSV.resolve()}")
    print(f"Saved: {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()
