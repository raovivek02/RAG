"""Download one openFDA drug label per drug in DRUGS into data/raw/."""
import json, time, requests
from pathlib import Path

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

DRUGS = [
    "metformin", "atorvastatin", "lisinopril", "amlodipine", "omeprazole",
    "levothyroxine", "losartan", "sertraline", "gabapentin", "warfarin",
    "ibuprofen", "amoxicillin", "azithromycin", "prednisone", "albuterol",
    "clopidogrel", "furosemide", "tramadol", "insulin glargine", "apixaban",
    "rivaroxaban", "empagliflozin", "semaglutide", "pembrolizumab",
    "adalimumab", "montelukast", "escitalopram", "pantoprazole",
    "simvastatin", "hydrochlorothiazide",
]

URL = "https://api.fda.gov/drug/label.json"

for name in DRUGS:
    fp = OUT / f"{name.replace(' ', '_')}.json"
    if fp.exists():
        continue
    params = {"search": f'openfda.generic_name:"{name}"', "limit": 1}
    try:
        r = requests.get(URL, params=params, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            print(f"[skip] no result for {name}")
            continue
        fp.write_text(json.dumps(results[0], indent=2))
        print(f"[ok] {name}")
    except Exception as e:
        print(f"[fail] {name}: {e}")
    time.sleep(0.5)