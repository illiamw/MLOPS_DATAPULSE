from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# main.py:
# MLOPS_DATAPULSE/
# └── src/
#     └── app_mlops/
#         └── main.py
#
# parents[0] -> app_mlops
# parents[1] -> src
# parents[2] -> MLOPS_DATAPULSE

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_data_raw.csv"
)

SERVING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "serving"
    / "industrial_data_serving.csv"
)


# ============================================================
# DEBUG PATHS
# ============================================================

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"SRC_ROOT: {SRC_ROOT}")
print(f"SCRIPTS_ROOT: {SCRIPTS_ROOT}")
print(f"RAW_DATA_PATH: {RAW_DATA_PATH}")
print(f"SERVING_DATA_PATH: {SERVING_DATA_PATH}")


# ============================================================
# PYTHON PATH
# ============================================================

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# INTERNAL IMPORTS
# ============================================================

from app_mlops.serving.inference import predict
import app_mlops.models.train as train
from app_mlops.data.load_data import load_data

from scripts.run_data_transform import main as transf_pipeline
from scripts.run_pipeline import main as run_pipeline

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="DataPulse - Industrial Prediction API",
    description="ML API for predicting failures in the industrial sector",
    version="1.0.0"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {"status": "ok_v01"}


# ============================================================
# REQUEST DATA SCHEMA
# ============================================================


class ModelRequest(BaseModel):
    model_run: str


class PredictionData(BaseModel):
    registro_id: str
    data_registro: str
    linha_producao: str
    turno: str
    maquina: str
    idade_maquina_anos: int
    temperatura_valor: float
    unidade_temperatura: str
    pressao_valor: float
    unidade_pressao: str
    vibracao_motor_mm_s: float
    velocidade_esteira_m_min: float
    umidade_pct: float
    tamanho_lote: int
    tempo_setup_min: float
    paradas_nao_planejadas: int
    taxa_defeitos_pct: float
    energia_sensor_b_kwh: float
    codigo_campanha: str
    ruido_aleatorio: float
    consumo_energia_kwh: float
    falha_24h: int


class PredictionRequest(BaseModel):
    model: ModelRequest
    data: PredictionData


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def get_prediction(requisition: PredictionRequest):
    """
    Receives industrial data and performs failure prediction.
    """

    try:

        # ----------------------------------------------------
        # 1. Load historical raw data
        # ----------------------------------------------------

        raw_data = load_data(
            RAW_DATA_PATH
        )

        # ----------------------------------------------------
        # 2. Add new prediction record
        # ----------------------------------------------------

        data = requisition.data

        raw_data = pd.concat(
            [
                raw_data,
                pd.DataFrame([data.model_dump()])
            ],
            ignore_index=True
        )

        # ----------------------------------------------------
        # 3. Data transformation pipeline
        # ----------------------------------------------------

        transformed_data = transf_pipeline(
            df=raw_data
        )

        # ----------------------------------------------------
        # 4. Keep only the new record
        # ----------------------------------------------------

        transformed_data = transformed_data.iloc[[-1]]

        print("###########################################################")
        print("Dados transformados:")
        print(transformed_data)

        # ----------------------------------------------------
        # 5. Model inference
        # ----------------------------------------------------

        result = predict(
            model_path=requisition.model.model_run,
            df=transformed_data
        )

        return {
            "prediction": result
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# ============================================================
# MODELS
# ============================================================

import os
import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

from fastapi import HTTPException


@app.get("/models")
def get_active_models():

    try:

        # ====================================================
        # MLflow
        # ====================================================

        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://mlflow:5000"
        )

        mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient(
            tracking_uri=tracking_uri
        )

        # ====================================================
        # EXPERIMENT
        # ====================================================

        experiment_id = "2"

        # ====================================================
        # LOGGED MODELS
        # ====================================================

       # Apenas Runs ativas (não deletadas)
        active_runs = client.search_runs(
            experiment_ids=[experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1000
        )

        active_run_ids = {
            run.info.run_id
            for run in active_runs
        }

        # Busca os Logged Models
        models = client.search_logged_models(
            experiment_ids=[experiment_id],
            max_results=100,
            order_by=[
                {
                    "field_name": "creation_time",
                    "ascending": False
                }
            ]
        )

        # Apenas modelos cuja Run de origem está ativa
        models = [
            model
            for model in models
            if model.source_run_id in active_run_ids
        ]
        # ====================================================
        # RESULT
        # ====================================================

        result = []

        for model in models:

            # -----------------------------------------------
            # Métricas do LoggedModel
            # -----------------------------------------------

            metrics = {
                metric.key: metric.value
                for metric in model.metrics
            }

            result.append({
                "model_id": model.model_id,
                "model_name": model.name,
                "run_id": model.source_run_id,
                "status": model.status,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "f1": metrics.get("f1")
            })

        return {
            "experiment": "PrevisaoFalha24h",
            "experiment_id": experiment_id,
            "total": len(result),
            "models": result
        }

    except Exception as e:

        print("========================================")
        print("ERROR /models")
        print("========================================")

        print(
            f"{type(e).__name__}: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Erro ao consultar modelos MLflow: "
                f"{type(e).__name__}: {e}"
            )
        )


    
# ============================================================
# Experiment execution endpoint
# ============================================================

@app.post("/allexperiments")
def run_allexperiments_endpoint():
    try:
        return train.run_allexperiments()
    except Exception as e:
        return {"error": str(e)}

@app.post("/experiment")
def run_experiment_endpoint(model: dict = None):
    try:
        train.run_experiment(model)
        return {"status": "Experiment executed successfully."} 
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# PIPELINE EXECUTION ENDPOINT
# ============================================================

@app.post("/pipeline")
def run_pipeline_endpoint():
    try:
        run_pipeline()
        return {"status": "run_pipeline executed successfully."} 
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting FastAPI server teste...")
    print("train.run_allexperiments =", train.run_allexperiments)
    print("module =", train.run_allexperiments.__module__)
    print("name =", train.run_allexperiments.__name__)

    train.run_allexperiments()

    print("End teste...")