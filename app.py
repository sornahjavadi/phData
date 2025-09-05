from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
import numpy as np
import pickle

MODEL_DIR = pathlib.Path(__file__).resolve().parent / "model"
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"

try:
    MODEL = pickle.load(open(MODEL_DIR / "model.pkl", "rb"))
except FileNotFoundError:
    raise RuntimeError(
        f"Model artefacts not found in {MODEL_DIR}.  Run 'python create_model.py' first."
    )

try:
    MODEL_FEATURES: List[str] = json.load(open(MODEL_DIR / "model_features.json"))
except FileNotFoundError:
    raise RuntimeError(
        f"Model feature specification not found.  Ensure model_features.json exists in {MODEL_DIR}."
    )

_demographics_path = DATA_DIR / "zipcode_demographics.csv"
if _demographics_path.exists():
    DEMOGRAPHICS_DF = pd.read_csv(_demographics_path, dtype={"zipcode": str})
else:
    DEMOGRAPHICS_DF = pd.DataFrame()


def enrich_with_demographics(df: pd.DataFrame) -> pd.DataFrame:
    if DEMOGRAPHICS_DF.empty:
        for col in set(MODEL_FEATURES) - set(df.columns):
            df[col] = np.nan
        return df
    if "zipcode" not in df.columns:
        raise HTTPException(status_code=422, detail="'zipcode' field is required for enrichment")
    df = df.copy()
    df["zipcode"] = df["zipcode"].astype(str)
    merged = df.merge(DEMOGRAPHICS_DF, how="left", on="zipcode")
    return merged


class FullHouseRecord(BaseModel):
    bedrooms: Optional[float] = Field(..., description="Number of bedrooms")
    bathrooms: Optional[float] = Field(..., description="Number of bathrooms")
    sqft_living: Optional[float] = Field(..., description="Living area in square feet")
    sqft_lot: Optional[float] = Field(..., description="Lot size in square feet")
    floors: Optional[float] = Field(..., description="Number of floors")
    waterfront: Optional[int] = Field(None, description="1 if the house is on the waterfront, 0 otherwise")
    view: Optional[int] = Field(None, description="Index of the view rating")
    condition: Optional[int] = Field(None, description="Condition rating of the house")
    grade: Optional[int] = Field(None, description="Grade of the house")
    sqft_above: Optional[float] = Field(..., description="Square footage of interior housing space above ground level")
    sqft_basement: Optional[float] = Field(..., description="Square footage of the basement")
    yr_built: Optional[int] = Field(None, description="Year the house was built")
    yr_renovated: Optional[int] = Field(None, description="Year the house was renovated")
    zipcode: Optional[str] = Field(..., description="ZIP code of the house location")
    lat: Optional[float] = Field(None, description="Latitude coordinate")
    long: Optional[float] = Field(None, description="Longitude coordinate")
    sqft_living15: Optional[float] = Field(None, description="Living area of the 15 nearest neighbors")
    sqft_lot15: Optional[float] = Field(None, description="Lot size of the 15 nearest neighbors")

    @validator("zipcode", pre=True)
    def _coerce_zip(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return str(v)


class RequiredHouseRecord(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: str

    @validator("zipcode", pre=True)
    def _coerce_zipcode(cls, v: Any) -> str:
        return str(v)


app = FastAPI(
    title="Housing Price Prediction API",
    description=(
        "An example service for predicting house prices in King County.\n"
        "Submit housing attributes in the request body to receive predicted\n"
        "sale prices.  Demographic data is automatically appended based on\n"
        "the provided ZIP code."
    ),
    version="1.0.0",
)


@app.post("/predict", summary="Predict house prices with all available features")
async def predict(records: Union[FullHouseRecord, List[FullHouseRecord]]):
    if not isinstance(records, list):
        record_list = [records]
    else:
        record_list = records
    data = [r.dict(exclude_unset=True) for r in record_list]
    df = pd.DataFrame(data)
    missing_cols = set(["bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                        "floors", "sqft_above", "sqft_basement", "zipcode"]) - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields for prediction: {sorted(missing_cols)}"
        )
    enriched = enrich_with_demographics(df)
    X = enriched.reindex(columns=MODEL_FEATURES)
    X_filled = X.fillna(0)
    X_array = X_filled.astype(float).to_numpy(copy=True)
    preds = MODEL.predict(X_array)
    return {
        "predictions": preds.tolist(),
    }


@app.post("/predict_required", summary="Predict prices with minimal house attributes")
async def predict_required(records: Union[RequiredHouseRecord, List[RequiredHouseRecord]]):
    if not isinstance(records, list):
        record_list = [records]
    else:
        record_list = records
    data = [r.dict() for r in record_list]
    df = pd.DataFrame(data)
    enriched = enrich_with_demographics(df)
    X = enriched.reindex(columns=MODEL_FEATURES)
    X_filled = X.fillna(0)
    X_array = X_filled.astype(float).to_numpy(copy=True)
    preds = MODEL.predict(X_array)
    return {
        "predictions": preds.tolist(),
    }


@app.get("/healthz", summary="Health check endpoint")
async def healthcheck():
    return {"status": "ok"}
