import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Create synthetic data if needed
def create_synthetic_data():
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'tenure': np.random.randint(1, 73, n_samples),
        'MonthlyCharges': np.random.uniform(18, 120, n_samples),
        'TotalCharges': np.random.uniform(18, 8700, n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No'], n_samples),
    }
    
    # Create target based on some logic
    churn_prob = (
        (data['tenure'] < 12) * 0.3 +
        (data['MonthlyCharges'] > 80) * 0.2 +
        (np.array(data['Contract']) == 'Month-to-month') * 0.3 +
        np.random.random(n_samples) * 0.2
    )
    data['Churn'] = (churn_prob > 0.5).astype(int)
    
    return pd.DataFrame(data)

def train_model():
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # Load or create data
    try:
        df = pd.read_csv('data/telco_churn.csv')
        # Convert TotalCharges to numeric if it exists
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df = df.dropna()
    except:
        print("Creating synthetic dataset...")
        df = create_synthetic_data()
        df.to_csv('data/telco_churn.csv', index=False)
    
    # Prepare features
    # Select numeric and categorical features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['Contract', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'TechSupport']
    
    # Handle missing columns
    available_numeric = [col for col in numeric_features if col in df.columns]
    available_categorical = [col for col in categorical_features if col in df.columns]
    
    # Prepare feature matrix
    X = df[available_numeric].copy()
    
    # Encode categorical variables
    le_dict = {}
    for col in available_categorical:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Target variable
    y = df['Churn'] if 'Churn' in df.columns else df.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and encoders
    joblib.dump(model, 'model/churn_model.pkl')
    joblib.dump(le_dict, 'model/label_encoders.pkl')
    joblib.dump(list(X.columns), 'model/feature_names.pkl')
    
    print("Model saved successfully!")
    return accuracy

if __name__ == "__main__":
    train_model()
