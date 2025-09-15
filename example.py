import requests

data = {
    "tenure": 12,
    "MonthlyCharges": 85.0,
    "TotalCharges": 1020.0,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())