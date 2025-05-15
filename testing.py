
import requests
url = "http://127.0.0.1:1234/invocations"
payload = {
    "dataframe_split": {
        "columns": ["High_Blood_Pressure", "High_Cholesterol", "CholCheck",
                    "BMI", "Smoker", "Stroke", "cardiovascular_disease", "PhysActivity",
                    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
                    "DiffWalk", "Gender", "Age", "Education", "Income"],
        "data": [
            [1.0, 1.0, 1.0, 40.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 9.0, 4.0, 3.0],
            [0.0, 0.0, 0.0, 25.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 7.0, 6.0, 1.0],
            [1.0, 1.0, 1.0, 28.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 9.0, 4.0, 8.0]
        ]
    }
}
response = requests.post(url, json=payload)
print("Response status:", response.status_code)
print("Response content:", response.json())

