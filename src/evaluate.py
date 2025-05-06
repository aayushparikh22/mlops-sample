import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate():
    df = pd.read_csv("data/sample_data.csv")
    X = df[['feature1', 'feature2']]
    y = df['label']
    model = joblib.load("model/model.pkl")
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Full dataset accuracy: {acc:.2f}")

if __name__ == "__main__":
    evaluate()
