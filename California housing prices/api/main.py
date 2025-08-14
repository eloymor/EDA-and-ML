import os
from typing import Optional, Dict, Any, List, Literal
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import pandas as pd
import xgboost as xgb
from sklearn.cluster import DBSCAN
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
import logging
from train_model import api_train
from etl import etl_api

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MODEL_PATH = os.path.join("..", "models", "xgboost_model.json")
DBSCAN_PATH = os.path.join("..", "models", "dbscan_model.joblib")
with open(os.path.join("..", "data", "proximity_mapping.json"), "r") as f:
    ocean_proximity_map: dict[str, int] = json.load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield

app = FastAPI(title="California Housing Price Prediction API",
              description="Predict median_house_value from California housing " +
                          "features using a pre-trained XGBoost model.",
              lifespan=lifespan)

booster: Optional[xgb.Booster] = None
feature_names: Optional[List[str]] = None


class PredictRequest(BaseModel):
    longitude: float = Field(..., description="Longitude of the location")
    latitude: float = Field(..., description="Latitude of the location")
    housing_median_age: float = Field(..., description="Median age of the housing units")
    total_rooms: float = Field(..., description="Total number of rooms")
    total_bedrooms: float = Field(..., description="Total number of bedrooms")
    population: float = Field(..., description="Total population")
    households: float = Field(..., description="Total households")
    median_income: float = Field(..., description="Median income of the households")
    ocean_proximity: str = Field(..., description="Ocean proximity to the location")

    @field_validator("ocean_proximity", mode="before")
    def validate_type(cls, v: str) -> str:
        low = v.strip().lower()
        if low not in {"<1h ocean", "inland", "near ocean", "near bay", "island"}:
            raise ValueError("type must be '<1h ocean', 'inland', 'near ocean', 'near bay', or 'island'")
        return low

    @field_validator("longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                      "population", "households", "median_income", mode="before")
    def validate_numeric(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError(f"{v} must be numeric")
        return v

def _load_model() -> None:
    global booster, feature_names
    if not os.path.exists(MODEL_PATH):
        booster = None
        feature_names = None
        return
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)
    booster = bst
    # feature_names are stored in Booster for models trained with DataFrame
    feature_names = getattr(bst, "feature_names", None)

def _proximity_to_numeric(v: str) -> int:
    proximity_map_lowercase = {k.lower(): v for k, v in ocean_proximity_map.items()}
    return proximity_map_lowercase.get(v.lower(), 1)


@app.get("/health")
async def health() -> Dict[str, Any]:
    exists = os.path.exists(MODEL_PATH)
    expected_features = list(getattr(PredictRequest, "model_fields", {}).keys())
    return {
        "status": "ok",
        "model_loaded": booster is not None,
        "model_file_exists": exists,
        "expected_features": expected_features,
    }

engine = create_engine('sqlite:///../etl_data.db')
Base = declarative_base()

class RawData(Base):
    __tablename__ = 'raw_data'
    index = Column(Integer, primary_key=True)
    longitude = Column(Float)
    latitude = Column(Float)
    housing_median_age = Column(Integer)
    total_rooms = Column(Integer)
    total_bedrooms = Column(Integer)
    population = Column(Integer)
    households = Column(Integer)
    median_income = Column(Float)
    median_house_value = Column(Float)
    ocean_proximity = Column(String)

    def to_json(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}


class ProcessedData(Base):
    __tablename__ = 'processed_data'
    index = Column(Integer, primary_key=True)
    housing_median_age = Column(Integer)
    total_rooms = Column(Integer)
    total_bedrooms = Column(Integer)
    population = Column(Integer)
    households = Column(Integer)
    median_income = Column(Float)
    median_house_value = Column(Float)
    ocean_proximity = Column(Integer)
    clusters = Column(Integer)

Session = sessionmaker(bind=engine)

def _retrieve_all_data(type: Literal["raw", "processed"] = "raw"):
    session = Session()
    logger.info("Retrieving data from database")
    data_dicts = []
    try:
        if type == "raw":
            data = session.query(RawData).all()
        elif type == "processed":
            data = session.query(ProcessedData).all()
        for item in data:
            data_dict = {column.name: getattr(item, column.name) for column in item.__table__.columns}
            data_dict.pop("index", None)
            if type == "raw":
                data_dict.pop("median_house_value", None)
            data_dicts.append(data_dict)
    except Exception as e:
        print(f"Error retrieving data: {e}")
    finally:
        session.close()
        logger.info("Data retrieval completed")
        return data_dicts

@app.get("/raw_data")
async def raw_data() -> JSONResponse:
    data_dicts = _retrieve_all_data("raw")
    return JSONResponse(content=data_dicts, status_code=200)

@app.get("/processed_data")
async def processed_data() -> JSONResponse:
    data_dicts = _retrieve_all_data("processed")
    return JSONResponse(content=data_dicts, status_code=200)

def _build_feature_frame(req: PredictRequest) -> pd.DataFrame:
    df = pd.DataFrame(
        data=[[
            req.longitude, req.latitude, req.housing_median_age, req.total_rooms, req.total_bedrooms,
            req.population, req.households, req.median_income, _proximity_to_numeric(req.ocean_proximity)
        ]],
        columns=list(getattr(req, "model_fields", {}).keys()) # PredictRequest class items
    )
    # Take care of the maximum value of housing_median_age set during training:
    df.loc[df["housing_median_age"] > 52, "housing_median_age"] = 52
    # Get all raw data, need this operation to perform DBSCAN clustering:
    raw_data = _retrieve_all_data()
    df = pd.concat([df, pd.DataFrame(raw_data)], axis=0)
    # Apply DBSCAN clustering for location:
    density_cluster = DBSCAN(eps=0.1, min_samples=5, metric="euclidean")
    clusters = density_cluster.fit_predict(df[["latitude", "longitude"]])
    df["clusters"] = clusters
    df = df.iloc[[0]] # Keep the first row only (coming from the request)
    df["ocean_proximity"] = df["ocean_proximity"].map(ocean_proximity_map)
    # Drop latitude and longitude columns:
    df.drop(columns=["latitude", "longitude"], inplace=True)
    if list(df.columns) != feature_names:
        raise ValueError("Unexpected features in request")

    return df


@app.post("/predict")
async def predict(payload: PredictRequest) -> JSONResponse:
    if booster is None:
        # Try lazy load once more
        _load_model()
    if booster is None:
        raise HTTPException(status_code=500, detail=f"Model not found at {MODEL_PATH}. Train and save the model first.")

    # Build feature frame aligned to model features
    X = _build_feature_frame(payload)

    try:
        dmatrix = xgb.DMatrix(X)
        y_pred = booster.predict(dmatrix)
        pred_value = round(float(y_pred[0]), 2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    final_result = {
        "prediction": pred_value,
        "model_features_used": feature_names,
    }
    return JSONResponse(content=final_result, status_code=200)


@app.get("/train")
@app.post("/train/{db_name}/{table_name}")
async def train_model(db_name: Optional[str] = None, table_name: str = None) -> JSONResponse:
    try:
        if db_name is None:
            db_name = "etl_data.db"
        if table_name is None:
            table_name = "processed_data"
        await api_train(db_name, table_name)
        return JSONResponse(content={"status": "Training completed"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.post("/etl/{file_path}")
@app.post("/etl/{file_path}/{db_name}/{table_name}")
async def etl(file_path: str, db_name: str = None, table_name: str = None) -> JSONResponse:
    if db_name is None:
        db_name = "etl_data.db"
    if table_name is None:
        table_name = "processed_data"
    try:
        await etl_api(file_path, db_name, table_name)
        return JSONResponse(content={"status": "ETL completed"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ETL failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
