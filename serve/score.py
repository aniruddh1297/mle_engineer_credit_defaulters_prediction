import json
import pandas as pd
import mlflow.pyfunc
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

def init():
    global model
    logger.info("🔁 Starting model initialization...")

    try:
        # 🔧 Correctly resolve path to the MLflow model directory
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
        logger.info(f"📦 Resolved model path: {model_path}")

        # ✅ Load the MLflow model
        model = mlflow.pyfunc.load_model(model_path)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def run(input_data):
    logger.info(f"📨 Received input: {input_data}")

    try:
        # 🔄 Parse input
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

        logger.info(f"📄 Converted input to DataFrame:\n{input_df}")

        # 🔍 Make predictions
        predictions = model.predict(input_df)
        logger.info(f"✅ Predictions: {predictions.tolist()}")
        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.error(f"❌ Inference error: {e}")
        return {"error": str(e)}
