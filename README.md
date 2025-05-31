# 🧠 Credit Card Default Prediction using Azure Machine Learning

## 📌 Overview

This project builds a **production-ready ML pipeline** to predict whether a credit card client will default on their payment next month. The solution uses **Azure Machine Learning**, implements **MLOps best practices**, and includes support for **deployment, CI/CD, explainability, and business impact metrics**.

## 💼 Business Problem

Defaulting credit card customers significantly impact financial institutions. Accurately identifying high-risk clients enables the bank to take preventive action. This model predicts defaults using customer behavior and credit profile.

## 🏗️ Architecture

- Azure ML SDK v2 + MLFlow
- Multi-component SDK pipeline: preprocess → train → evaluate
- Cross-environment setup: dev, test, prod workspaces
- Model promotion strategy from test → prod
- Automated deployment via Azure Online Endpoints
- CI/CD with GitHub Actions (includes conditional triggers for dev/test/prod)
- Environment and component registration handled programmatically

## 📁 Project Structure

```
.
├── .github/workflows/            # CI/CD workflows
├── data/                         # Raw dataset
├── components/                   # Azure ML components
│   ├── preprocess_component.py
│   ├── train_component.py
│   └── evaluate_component.py
├── pipelines/                    # Run & register scripts
│   ├── run_pipeline.py
│   └── promote_model.py
├── serve/                        # Deployment artifacts
│   ├── score.py
│   ├── inference_config.yaml
│   ├── deploy_endpoint.py
│   └── sample_request.json
├── environments/                 # Conda environment YAMLs
├── config/                       # Workspace config files
├── register_scripts/            # Environment/component promotion
│   ├── register_env.py
│   ├── register_component.py
│   └── promote_model.py
├── requirements.txt
└── README.md
```

## 🧪 ML Pipeline Components

- 🔹 **Preprocessing**: Scales data, handles missing values
- 🔹 **Training**: Trains XGBoost, Logistic Regression, Random Forest; logs best model via MLflow
- 🔹 **Evaluation**:
  - Classification metrics (F1, ROC-AUC, Precision/Recall)
  - Cost-sensitive threshold optimization
  - SHAP explainability
  - Confusion matrix & curves logged to MLflow

## 🚀 Deployment

The best model is deployed via Azure ML Online Endpoint:

- `score.py`: Inference script using MLflow
- `deploy_endpoint.py`: Script to deploy endpoint
- `inference_config.yaml`: Configuration for endpoint creation
- `sample_request.json`: Test payload

You can invoke the endpoint using:
```bash
curl -X POST <ENDPOINT_URL> -H "Authorization: Bearer <TOKEN>" -d @sample_request.json
```

## 🔁 Model Promotion Strategy

- Models are registered in the **test workspace** and evaluated.
- If they pass performance thresholds, they are:
  - Promoted via `promote_model.py` to the **prod workspace**
  - Automatically deployed via CI/CD if the commit message includes `deploy to prod`

> 🔒 NOTE: The model is currently not tagged as `validated`. This is a planned future enhancement.

## ✅ CI/CD with GitHub Actions

CI/CD workflow automates:
- Azure ML environment & component registration
- Triggering the training pipeline
- Promoting the best model to prod
- Deploying the model to a managed online endpoint

Customizable triggers:
- Commit message with `pipeline-start`: runs pipeline even on `dev`
- Commit message with `prod-run`: forces full pipeline in `main` (prod)
- Otherwise, promotion from test to prod `main`
- As for Test env , the pipeline will always run keeping in mind reproducibility and verification in mind

CI/CD Workflow:
```yaml
# Azure ML Multi-Env Pipeline CI
# (see .github/workflows/azureml-pipeline-ci.yml)

on:
  push:
    branches:
      - release/dev
      - release/test
      - main
    paths:
      - "component_code/**.yaml"
      - "component_code/**.py"
      - "pipeline/run_pipeline.py"
      - "register_scripts/**.py"
      - "promote_model.py"
      - "config/**.json"
      - "requirements.txt"
      - ".github/workflows/azureml-pipeline-ci.yml"
```

## 📈 Model Performance

| Metric       | Value   |
|--------------|---------|
| Accuracy     | 77.1%   |
| F1 Score     | 55.1%   |
| ROC-AUC      | 76.5%   |
| Precision    | 48.6%   |
| Recall       | 56.8%   |
| Optimal Threshold | 0.69 |
| Estimated Cost | ~764K |

All metrics are logged to MLflow.

## 🧠 Explainability

SHAP values are computed and logged during evaluation. Beeswarm plots highlight most influential features in predictions. Saved as `shap_beeswarm.png` and logged to MLflow.

## 🔒 Cost & Security

- Cost-sensitive thresholding built into evaluation.
- Small instance types used (e.g., `Standard_E2s_v3`).
- Credentials managed securely via Azure CLI & GitHub Secrets.

## 📎 How to Run Locally

1. Clone the repo
2. Set up environment:
   ```bash
   pip install -r requirements.txt
   az login
   ```
3. Run:
   ```bash
   python register_scripts/register_env.py --env dev
   python register_scripts/register_component.py --env dev
   python pipelines/run_pipeline.py --env dev
   ```

To deploy:
```bash
cd serve
python deploy_endpoint.py
```

## 🧪 Endpoint Testing

Use `sample_request.json` and test with:
```bash
curl -X POST <URL> -H "Authorization: Bearer <TOKEN>" -d @sample_request.json
```

## 🙌 Acknowledgments

- Azure ML SDK v2
- MLOps best practices
- MLflow for experiment tracking
- scikit-learn, XGBoost

#test