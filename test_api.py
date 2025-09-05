import json
from typing import List
import pandas as pd
import requests

BASE_URL = "http://localhost:8000"

def main() -> None:
    examples = pd.read_csv("data/future_unseen_examples.csv").iloc[:3]
    payload_full: List[dict] = examples.to_dict(orient="records")
    resp = requests.post(f"{BASE_URL}/predict", json=payload_full)
    print("Status:", resp.status_code)
    print("Full endpoint predictions:")
    print(json.dumps(resp.json(), indent=2))
    minimal_cols = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "sqft_above", "sqft_basement", "zipcode"
    ]
    payload_min = examples[minimal_cols].astype({"zipcode": str}).to_dict(orient="records")
    resp_min = requests.post(f"{BASE_URL}/predict_required", json=payload_min)
    print("\nMinimal endpoint predictions:")
    print(json.dumps(resp_min.json(), indent=2))

if __name__ == "__main__":
    main()
