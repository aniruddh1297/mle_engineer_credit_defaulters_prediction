# import json
# import joblib
# import numpy as np
# import pandas as pd
# from typing import Any

# def init():
#     global model
#     model = joblib.load("model.pkl")
#     model = joblib.load(model_path)
#     print("✅ Model loaded successfully")

# def run(raw_data: Any):
#     try:
#         data = json.loads(raw_data)
#         input_df = pd.DataFrame(data)

#         # Optional: Validate input columns if needed
#         required_features = model.feature_names_in_  # requires sklearn >= 1.0
#         input_df = input_df[required_features]

#         probabilities = model.predict_proba(input_df)[:, 1]
#         predictions = (probabilities > 0.45).astype(int)  # Use your tuned threshold

#         return json.dumps({
#             "probabilities": probabilities.tolist(),
#             "predictions": predictions.tolist()
#         })
#     except Exception as e:
#         error_msg = f"❌ Error during prediction: {str(e)}"
#         print(error_msg)
#         return json.dumps({"error": error_msg})


# serve/score.py

import json
import pandas as pd
import mlflow.pyfunc
import logging
from azureml.core.model import Model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

def init():
    global model
    logger.info("🔁 Starting model initialization...")

    try:
        # Get the path to the registered model
        model_path = Model.get_model_path("credit-default-model")
        logger.info(f"📦 Resolved model path: {model_path}")

        # Load the MLflow model
        model = mlflow.pyfunc.load_model(model_path)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def run(input_data):
    logger.info(f"📨 Received input: {input_data}")

    try:
        # Parse input if it is a JSON string
        if isinstance(input_data, str):
            input_data = json.loads(input_data)

        # Convert input to pandas DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise ValueError("Unsupported input format type.")

        logger.info(f"📄 Converted input to DataFrame:\n{input_df}")

        # Predict
        predictions = model.predict(input_df)
        logger.info(f"✅ Predictions: {predictions.tolist()}")
        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.error(f"❌ Inference error: {e}")
        return {"error": str(e)}
