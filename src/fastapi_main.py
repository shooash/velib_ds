import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi_connector import FastAPIConnector
# from general import DataFiles, log

app = FastAPI()
connector = FastAPIConnector(model_name="best_mlp", run_id="best")

class PredictionRequest(BaseModel):
    station: str | list[str]
    date: str  | list[str]

class DayPredictionRequest(BaseModel):
    date: str | list[str]

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predict bike delta for a specific station and date/time.
    """
    return await asyncio.to_thread(connector.predict, request.station, request.date)

@app.post("/predict_day")
async def predict_day(request: DayPredictionRequest):
    """
    Predict bike delta for all stations for a given day.
    """
    return await asyncio.to_thread(connector.predict_day, request.date)

@app.post("/predict_station_day")
async def predict_station_day(request: PredictionRequest):
    """
    Predict bike delta for all stations for a given day.
    """
    return await asyncio.to_thread(connector.predict_station_day, request.station, request.date)

@app.post("/get_stations")
async def get_stations():
    """
    Get the list of Vélib stations.
    """
    return await asyncio.to_thread(connector.get_stations)


@app.post("/admin/retrain")
async def retrain():
    """
    Retrain the model with recent data.
    """
    return await asyncio.to_thread(connector.retrain)

@app.post("/admin/refresh")
async def refresh():
    """
    Update the datasets.
    """
    return await asyncio.to_thread(connector.refresh)


@app.get("/health")
async def health():
    """
    Check API health.
    """
    return {"status": "ok"}