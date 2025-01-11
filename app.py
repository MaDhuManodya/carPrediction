import joblib
import numpy as np

# Load the saved model
rfc_loaded = joblib.load('random_forest_model.pkl')

# Example input data (ensure this matches the feature set used in training)
# For example, the model might expect features like [Engine size, Weight, Safety rating, etc.]
# You need to replace this with the actual input data you're using
input_data = np.array([[1500, 1200, 3, 8]])  # Replace with your car's feature values

# Predict using the loaded model
prediction = rfc_loaded.predict(input_data)

# Output the predicted result
if prediction == 1:
    print("The car is safe.")
else:
    print("The car is not safe.")
