# MLflow ML Lifecycle 

The project builds and compares multiple classification models to predict Type 2 Diabetes
using MLflow for experiment tracking and model management, and Hyperopt for hyperparameter tuning.

This repository demonstrates the usage of **MLflow** to manage the complete ML lifecycle including:
- Experiment Tracking
- Model Logging
- Model Evaluation
- Model Registry
  
## 🔧 Technologies Used
- Python
- Scikit-learn
- Pandas
- MLflow

## The script will:
- Train baseline models
- Run hyperparameter optimization for each
- Log metrics and parameters to MLflow
- Register and promote the best model to production in the MLflow model registry

## Mlflow tracking:
- Visit http://127.0.0.1:5000 in your browser to explore:
        Run comparisons
        Parameter tracking
        Metrics (accuracy, precision, recall, F1 score)
        Model artifacts
  
## 🚀 How to Run
```bash
1. Clone the repo:
git clone https://github.com/Suhakh/T2D_Mlflow.git
cd T2D_Mlflow

2. install requirements:
run  pip install -r requirements.txt

3. Start MLflow Tracking Server:
run mlflow ui

4. Run the Training Script:
run python main.py




