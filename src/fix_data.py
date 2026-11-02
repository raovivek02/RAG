"""Re-download labels where the first search returned combination products, using exact generic names."""
import json, time, requests
from pathlib import Path

FIX = {  # file stem -> exact generic name in openFDA
    "metformin": "METFORMIN HYDROCHLORIDE",
    "lisinopril": "LISINOPRIL",
    "hydrochlorothiazide": "HYDROCHLOROTHIAZIDE",
    "empagliflozin": "EMPAGLIFLOZIN",
    "amoxicillin": "AMOXICILLIN",
}
for stem, name in FIX.items():
    r = requests.get("https://api.fda.gov/drug/label.json",
                     params={"search": f'openfda.generic_name.exact:"{name}"', "limit": 1},
                     timeout=30)
    res = r.json().get("results", []) if r.ok else []
    if not res:
        print(f"[no exact match] {name}")
        continue
    Path(f"data/raw/{stem}.json").write_text(json.dumps(res[0], indent=2))
    print(f"[ok] {stem} -> {res[0]['openfda']['generic_name'][0]}")
    time.sleep(0.5)
