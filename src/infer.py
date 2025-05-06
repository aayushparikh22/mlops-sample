import joblib
import pandas as pd

def infer():
    model = joblib.load("model/model.pkl")
    sample = pd.DataFrame({"feature1": [0.5], "feature2": [-0.1]})
    prediction = model.predict(sample)
    print(f"Prediction for {sample.values.tolist()[0]}: {prediction[0]}")

if __name__ == "__main__":
    infer()
