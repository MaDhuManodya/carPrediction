import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Function to validate input features
def validate_features(input_df: pd.DataFrame, expected_features: list):
    missing_features = set(expected_features) - set(input_df.columns)
    extra_features = set(input_df.columns) - set(expected_features)
    if missing_features or extra_features:
        raise ValueError(
            f"Feature mismatch. Missing: {missing_features}, Extra: {extra_features}"
        )

# Function to predict safety
def predict_safety(input_params: dict) -> str:
    selected_features = ['buying', 'doors', 'lug_boot', 'maint', 'persons']
    input_df = pd.DataFrame([input_params])
    input_df_selected = input_df[selected_features]
    validate_features(input_df_selected, selected_features)
    prediction = model.predict(input_df_selected)
    return "Safe" if prediction[0] == 1 else "Unsafe"

# Example usage
if __name__ == "__main__":
    input_parameters = {
        'buying': 1,
        'doors': 4,
        'lug_boot': 1,
        'maint': 2,
        'persons': 2,
    }
    try:
        prediction_result = predict_safety(input_parameters)
        print("Prediction result:", prediction_result)
    except ValueError as e:
        print("Error:", e)
