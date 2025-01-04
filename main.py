import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List
from colorama import init
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

init(autoreset=True)

class DataFrameConverter:
    # Converts specified categorical columns in a DataFrame to numeric codes
    def __init__(self, df: pd.DataFrame):
        self.dataframe = df

    def convert(self, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].astype('category').cat.codes
            else:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        return self.dataframe

def correlation(CDF: pd.DataFrame, threshold: float = 0.6):
    # Identifies highly correlated features above a given threshold
    corr_matrix = CDF.corr()
    high_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for row in high_corr.columns:
        for col in high_corr.index:
            if abs(high_corr.loc[col, row]) > threshold:
                return f"High correlation between '{col}' and '{row}' with value {round(high_corr.loc[col, row], 2)}."
    return "No high correlations found."

def outliers(CDF: pd.DataFrame, pot_out_cols: List[str] = None):
    # Identifies outliers in specified columns using the IQR method
    if pot_out_cols is None:
        pot_out_cols = ['Monthly_Charges', 'CLTV']

    results = {}
    for col in pot_out_cols:
        if col in CDF.columns:
            Q1 = CDF[col].quantile(0.25)
            Q3 = CDF[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = CDF[(CDF[col] < lower) | (CDF[col] > upper)]
            if not outliers.empty:
                results[col] = outliers

    return results if results else "No outliers found."

def feature_engineering(df: pd.DataFrame, extra: bool = False) -> pd.DataFrame:
    # Prepares the dataset by dropping unnecessary columns and converting categorical features
    df = df.drop(columns=['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip_Code', 'Lat_Long',
                          'Latitude', 'Longitude', 'Churn_Value', 'Total_Charges', 'Churn_Score',
                          'Churn_Reason'])

    # Identify categorical columns for conversion
    cat_cols = [col for col in df.columns if len(df[col].unique()) <= 5] + ['Tenure_Category']
    CDF = DataFrameConverter(df).convert(cat_cols)

    if extra:
        correlation(CDF)  # Check correlations if extra analysis is required
        outliers(CDF)     # Identify outliers if extra analysis is required

    return CDF

def train(CDF: pd.DataFrame):
    # Splits data into train and test sets, trains the model, and selects important features
    X = CDF.drop(columns=['Churn_Label'])
    y = CDF['Churn_Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initial model training
    model = RandomForestClassifier(random_state=42, n_estimators=145, class_weight='balanced')
    model.fit(X_train, y_train)

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Select features with importance above threshold
    selected_features = feature_importance[feature_importance['Importance'] > 0.02]['Feature']
    X_train_reduced = X_train[selected_features]
    X_test_reduced = X_test[selected_features]

    # Retrain model with selected features
    model.fit(X_train_reduced, y_train)
    return model, X_test_reduced, y_test

def predict(model, X_test, y_test):
    # Generates predictions and calculates accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

def main():
    # Main workflow: load data, preprocess, train, and evaluate model
    path = './data/churn_data.csv'
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(' ', '_')

    # Create tenure categories and drop original tenure column
    bins = [-float('inf'), 6, 12, 24, 36, 48, float('inf')]
    labels = ['0-6 months', '6-12 months', '1-2 years', '2-3 years', '3-4 years', '4+ years']
    df['Tenure_Category'] = pd.cut(df['Tenure_Months'], bins=bins, labels=labels)
    df = df.drop(columns=['Tenure_Months'])

    processed_df = feature_engineering(df)
    model, X_test, y_test = train(processed_df)
    predict(model, X_test, y_test)

if __name__ == '__main__':
    main()