# 🧠 Credit Card Default Prediction using Azure Machine Learning

## 📌 Overview

This project builds a **production-grade ML pipeline** to predict whether a credit card client will default on their next payment. The solution uses **Azure Machine Learning SDK v2**, applies **MLOps best practices**, and supports **CI/CD, explainability, automated deployment**, and **cost-sensitive evaluation**.

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
├── .azureml/                         
│   ├── config.dev.json
│   ├── config.test.json
│   └── config.prod.json
├── .github/workflows/               
│   └── azureml-pipeline-ci.yml
├── component_code/                  
│   ├── evaluate/
│   ├── preprocess/
│   └── train/
├── config/                          
│   ├── compute.yaml
│   └── environment.yaml
├── data/                            
├── doc/                             
├── pipeline/                        
│   └── run_pipeline.py
├── promote_scripts/                
│   └── promote_model.py
├── register_scripts/               
│   ├── data_upload.py
│   ├── register_component.py
│   ├── register_compute.py
│   └── register_env.py
├── serve/                           
├── src(for_local_test)/            
├── utils/                           
├── .env                             
├── LICENSE
├── README.md
└── requirements.txt                 
```

## 🧪 ML Pipeline Components

- **Preprocessing**: Scales data, handles missing values
- **Training**: Trains XGBoost, Logistic Regression, Random Forest; logs best model via MLflow
- **Evaluation**:
  - Classification metrics (F1, ROC-AUC, Precision/Recall)
  - Cost-sensitive threshold optimization
  - SHAP explainability
  - Confusion matrix & curves logged to MLflow

## 🚀 Deployment

- `score.py`: Inference script using MLflow
- `deploy_endpoint.py`: Script to deploy endpoint
- `inference_config.yaml`: Configuration for endpoint creation
- `sample_request.json`: Test payload

```bash
curl -X POST <ENDPOINT_URL> -H "Authorization: Bearer <TOKEN>" -d @sample_request.json
```

## 🔁 Model Promotion Strategy

- Models are registered in the **test workspace** and evaluated.
- If they pass performance thresholds:
  - Promoted via `promote_model.py` to **prod**
  - Deployed automatically if `prod-run` is used in commit message

## ✅ CI/CD with GitHub Actions

Automates:
- Component and environment registration
- Pipeline trigger (conditional)
- Model promotion and deployment

Trigger conditions:
- `pipeline-start`: forces dev pipeline run
- `prod-run`: full pipeline run and deploy in prod
- Otherwise:
  - Dev: no-op
  - Test: always runs
  - Main: only promotes

```yaml
on:
  push:
    branches:
      - release/dev
      - release/test
      - main
```

## 📈 Model Performance

| Metric       | Value   |
|--------------|---------|
| Accuracy     | 77.1%   |
| F1 Score     | 55.1%   |
| ROC-AUC      | 76.5%   |
| Precision    | 48.6%   |
| Recall       | 56.8%   |
| Threshold    | 0.69    |
| Cost Estimate| ~€764K  |

## 🧠 Explainability

SHAP values highlight feature contributions. Logged:
- `shap_beeswarm.png`
- ROC/PR curves
- Confusion matrix heatmap

## 🔒 Cost & Security

- Threshold optimized using cost matrix (FP: €1,000, FN: €900)
- Uses small instances (`Standard_E2s_v3`)
- Secure token-based deployment via Azure and GitHub secrets

## ▶️ Running Locally

```bash
pip install -r requirements.txt
az login
```

```bash
python register_scripts/register_env.py --env dev
python register_scripts/register_component.py --env dev
python register_scripts/data_upload.py --env dev
python pipeline/run_pipeline.py --env dev
```

To deploy:
```bash
cd serve
python deploy_endpoint.py --env prod
```

## 🧪 Endpoint Test

```bash
curl -X POST <URL> -H "Authorization: Bearer <TOKEN>" -d @serve/sample_request.json
```

## 🙌 Acknowledgments

- Azure Machine Learning SDK v2
- MLflow for experiment tracking
- GitHub Actions for DevOps automation
- scikit-learn, XGBoost, SHAP