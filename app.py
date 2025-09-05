"""
Machine Learning Model Prediction API
------------------------------------

This FastAPI application exposes a RESTful interface for serving housing
price predictions using a pre‑trained scikit‑learn model.  The model was
trained on a subset of the King County housing dataset augmented with
demographic data by ZIP code.  Two endpoints are provided:

1. **/predict** – accepts the full set of input features from the
   ``future_unseen_examples.csv`` file (all columns except price, date and
   id) and returns price predictions.  The service automatically joins
   demographic attributes on the provided ZIP code.

2. **/predict_required** – accepts only the subset of house attributes
   required by the baseline model (bedrooms, bathrooms, sqft_living,
   sqft_lot, floors, sqft_above, sqft_basement and zipcode).  The
   demographic attributes are still appended on the backend before the
   prediction is made.

Both endpoints accept either a single record or a list of records and
return a list of predictions along with a copy of the features used to
generate each prediction.  If demographic information is missing for a
particular ZIP code, the fields are filled with ``None`` and the model
will emit a prediction based on whatever information is available.

To run the API locally: ``uvicorn app:app --reload`` then POST your
JSON payloads to ``http://localhost:8000/predict`` or
``/predict_required``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
import numpy as np

import pickle


# ---------------------------------------------------------------------------
# Load trained model and metadata
#
# The model and the order of its expected input features are saved under
# ``model/`` when running ``python create_model.py``.  We load these
# artefacts once at module import so subsequent requests reuse the same
# objects.  This avoids the overhead of re‑loading the model for every
# prediction and ensures thread safety in the FastAPI environment.
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

# Load demographic data for ZIP code enrichment.  Treat ZIP codes as strings
# to preserve leading zeros (if present).  On first import this will read
# from disk; subsequent imports within the same process reuse the cached
# DataFrame.
_demographics_path = DATA_DIR / "zipcode_demographics.csv"
if _demographics_path.exists():
    DEMOGRAPHICS_DF = pd.read_csv(_demographics_path, dtype={"zipcode": str})
else:
    DEMOGRAPHICS_DF = pd.DataFrame()


def enrich_with_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Merge input housing data with demographic data on ``zipcode``.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing at least a 'zipcode' column.  The column is
        coerced to ``str`` before merging to avoid accidental type mismatches.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with demographic columns appended.  If the
        demographics file is not found or the ZIP code is missing, the
        demographic columns will be added with NaN values.
    """
    # Early exit if we don't have demographics available
    if DEMOGRAPHICS_DF.empty:
        # fill missing demographic columns with NaN values
        for col in set(MODEL_FEATURES) - set(df.columns):
            df[col] = np.nan
        return df

    # Coerce zipcode to string for the join
    if "zipcode" not in df.columns:
        raise HTTPException(status_code=422, detail="'zipcode' field is required for enrichment")
    df = df.copy()
    df["zipcode"] = df["zipcode"].astype(str)
    merged = df.merge(DEMOGRAPHICS_DF, how="left", on="zipcode")
    return merged


# ---------------------------------------------------------------------------
# Pydantic models for request validation
#

class FullHouseRecord(BaseModel):
    """
    Schema for the full set of inputs corresponding to one record from
    ``future_unseen_examples.csv``.  All fields are optional to allow
    flexibility in partial updates; however, leaving required model fields
    blank will result in a validation error at prediction time.
    """

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

    # Coerce numeric ZIP codes to strings to retain leading zeros and satisfy
    # model expectations.
    @validator("zipcode", pre=True)
    def _coerce_zip(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return str(v)


class RequiredHouseRecord(BaseModel):
    """
    Schema for the minimal input record required by the baseline model.  Only
    the features used during training are defined here.  Additional fields
    will be ignored.
    """

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


# ---------------------------------------------------------------------------
# FastAPI application
#

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
    """
    Generate price predictions for one or more full house records.

    The payload may be a single JSON object or a list of objects.  Each
    object corresponds to one house listing with the attributes defined in
    :class:`FullHouseRecord`.

    Returns a list of predictions in the order the records were submitted.
    """
    # Normalise to a list for unified processing
    if not isinstance(records, list):
        record_list = [records]
    else:
        record_list = records

    # Convert list of pydantic models to DataFrame
    data = [r.dict(exclude_unset=True) for r in record_list]
    df = pd.DataFrame(data)
    # Ensure that at least the required baseline columns exist
    missing_cols = set(["bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                        "floors", "sqft_above", "sqft_basement", "zipcode"]) - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields for prediction: {sorted(missing_cols)}"
        )

    enriched = enrich_with_demographics(df)
    # Extract model features.  Missing demographic columns will be NaN
    X = enriched.reindex(columns=MODEL_FEATURES)
    # Replace any missing values with zero.  The baseline model does not
    # support NaN values so this simple imputation ensures a numeric
    # feature matrix.  In production one might instead use the mean or
    # median computed from the training set or incorporate an imputer in
    # the training pipeline.
    X_filled = X.fillna(0)
    # Convert to numpy array to decouple from the DataFrame.  ``copy=True``
    # creates a contiguous array to avoid potential threading issues.
    X_array = X_filled.astype(float).to_numpy(copy=True)
    preds = MODEL.predict(X_array)
    # Return predictions along with optional metadata (e.g. original order)
    return {
        "predictions": preds.tolist(),
    }


@app.post("/predict_required", summary="Predict prices with minimal house attributes")
async def predict_required(records: Union[RequiredHouseRecord, List[RequiredHouseRecord]]):
    """
    Generate price predictions when only the minimal feature set is provided.

    This endpoint accepts a single record or a list of records containing only
    the features that were used during training.  Demographic data is
    appended on the backend.
    """
    # Normalise to a list
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
    """Simple health check endpoint used for monitoring."""
    return {"status": "ok"}