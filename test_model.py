# test_model.py
# A simple script to test the model with manual inputs.
# Run this after training the model to see example recommendations.
# Make sure 'category_model.joblib', 'encoder.joblib', and 'categories.pkl' are in the same directory.

import joblib
import pandas as pd
import numpy as np

# Load the trained model, encoder, and categories
model = joblib.load('category_model.joblib')
enc = joblib.load('encoder.joblib')
categories = joblib.load('categories.pkl')

# Example conditions (change these as needed)
day = 'Mon'  # e.g., 'Mon', 'Tue', etc.
session = 'Dinner'  # 'Lunch' or 'Dinner'
weather = 'Rain'  # 'Rain', 'Wind', 'Cloud', 'Sunny'
waiter = 'Jim'  # e.g., 'Jim', 'Dwight', 'Toby', etc.

# Create input DataFrame
input_df = pd.DataFrame({
    'Day': [day],
    'Session': [session],
    'Weather': [weather],
    'Waiter': [waiter]
})

# Transform with encoder
X_input = enc.transform(input_df)

# Predict quantities per category
preds = model.predict(X_input)[0]

# Map predictions to categories
cat_preds = {cat: preds[i] for i, cat in enumerate(categories)}

# Sort by predicted quantity descending and take top 3
sorted_cats = sorted(cat_preds.items(), key=lambda x: x[1], reverse=True)[:3]

# Print recommendations with +20% target
print(f"Recommendations for {waiter} on {day}, {session}, Weather: {weather}")
for cat, pred_qty in sorted_cats:
    target_qty = int(round(pred_qty * 1.2))
    print(f"- Category: {cat}")
    print(f"  Predicted Quantity: {round(pred_qty, 2)}")
    print(f"  Target Quantity: {target_qty}")
    print()