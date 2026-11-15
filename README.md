# Churn Prediction Project  

This project predicts customer churn using a machine learning pipeline. It processes customer data, selects relevant features, and applies a Random Forest Classifier to classify customers as churned or retained.

## Features  
- **Data Preprocessing**: Handles categorical data and identifies outliers.  
- **Feature Engineering**: Drops irrelevant columns and selects significant features.  
- **Model Training**: Trains a Random Forest model with feature importance analysis.  
- **Evaluation**: Assesses model performance and accuracy.


## Run it

```bash
pip install -r requirements.txt
python main.py
```

Prints test-set accuracy for the Random Forest (about 0.78 with `random_state=42`). The data is the public IBM Telco customer churn dataset in `data/`.
