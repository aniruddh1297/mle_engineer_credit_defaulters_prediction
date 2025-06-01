import json
import pandas as pd
import mlflow.pyfunc
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

def init():
    global model
    logger.info("Starting model initialization...")

    try:
        base_model_dir = os.getenv("AZUREML_MODEL_DIR")
        logger.info(f"Base model directory: {base_model_dir}")

        for root, dirs, files in os.walk(base_model_dir):
            if "MLmodel" in files:
                model_path = root
                break
        else:
            raise FileNotFoundError("❌ Could not find 'MLmodel' in any subdirectories.")

        logger.info(f"📦 Resolved model path: {model_path}")
        model = mlflow.pyfunc.load_model(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
        
def run(input_data):
    logger.info(f"Received input: {input_data}")

    try:
        if isinstance(input_data, str):
            input_data = json.loads(input_data)

        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise ValueError("Unsupported input format type.")

        logger.info(f"Converted input to DataFrame:\n{input_df}")

        predictions = model.predict(input_df)
        logger.info(f"Predictions: {predictions.tolist()}")
        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {"error": str(e)}

