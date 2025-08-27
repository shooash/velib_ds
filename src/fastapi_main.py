from fastapi import FastAPI
from pydantic import BaseModel
from fastapi_connector import FastAPIConnector
# from general import DataFiles, log

app = FastAPI()
connector = FastAPIConnector(model_name="best_mlp", run_id="best")

class PredictionRequest(BaseModel):
    station: str
    date: str  

class DayPredictionRequest(BaseModel):
    date: str 

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predict bike delta for a specific station and date/time.
    """
    return connector.predict(request.station, request.date)

@app.post("/predict_day")
async def predict_day(request: DayPredictionRequest):
    """
    Predict bike delta for all stations for a given day.
    """
    return connector.predict_day(request.date)

@app.post("/admin/retrain")
async def retrain():
    """
    Retrain the model with recent data.
    """
    return connector.retrain()

@app.post("/admin/refresh")
async def refresh():
    """
    Update the datasets.
    """
    return connector.refresh()

@app.post("/admin/grid")
async def grid():
    """
    Train models to select best params.
    """
    return connector.grid()


@app.get("/health")
async def health():
    """
    Check API health.
    """
    return {"status": "ok"}