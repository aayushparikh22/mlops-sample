import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

import os



def train():
    df = pd.read_csv("data/sample_data.csv")
    X = df[['feature1', 'feature2']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model accuracy: {score:.2f}")
    os.makedirs("model", exist_ok=True) 
    joblib.dump(model, "model/model.pkl")

if __name__ == "__main__":
    train()
