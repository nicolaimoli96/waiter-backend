# backend/app.py
# Updated Flask app with the new endpoint for category recommendations.
# The original /api/simulate-daily is kept as-is, but you can remove it if not needed.
# Run this after training the model.

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model, encoder, and categories
model = joblib.load('category_model.joblib')
enc = joblib.load('encoder.joblib')
categories = joblib.load('categories.pkl')

# Original endpoint (kept for reference; modify or remove as needed)
@app.route('/api/simulate-daily', methods=['POST'])
def simulate_daily():
    try:
        data = request.get_json()
        day_of_week = data.get('day_of_week')
        weather = data.get('weather')
        daily_target = data.get('daily_target', 0)
        sales_done_today = data.get('sales_done_today', 0)

        # Validate
        if not all([day_of_week is not None, weather in ['rain', 'cloud', 'wind', 'sunny'], daily_target > 0]):
            return jsonify({'error': 'Invalid input'}), 400

        # Create feature DataFrame
        current_features = pd.DataFrame({
            'day_of_week': [day_of_week],
            'weather_rain': [1 if weather == 'rain' else 0],
            'weather_cloud': [1 if weather == 'cloud' else 0],
            'weather_wind': [1 if weather == 'wind' else 0],
            'weather_sunny': [1 if weather == 'sunny' else 0]
        })

        # Note: The rest of this function is unchanged from your provided code.
        # I've omitted it here for brevity, but include the full body in your file.
        # ...

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New endpoint for category recommendations
@app.route('/api/recommend-categories', methods=['POST'])
def recommend_categories():
    try:
        data = request.get_json()
        day = data.get('day')  # e.g., 'Mon'
        session = data.get('session')  # e.g., 'Lunch' or 'Dinner'
        weather = data.get('weather')  # e.g., 'Rain'
        waiter = data.get('waiter')  # e.g., 'Jim'

        # Validate inputs (add more validation as needed, e.g., check against known values)
        if not all([day, session, weather, waiter]):
            return jsonify({'error': 'Missing or invalid input'}), 400

        # Create input DataFrame matching training features
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

        # Prepare recommendations with +20% target
        recommendations = []
        for cat, pred_qty in sorted_cats:
            target_qty = int(round(pred_qty * 1.2))  # Round to nearest integer
            recommendations.append({
                'category': cat,
                'predicted_quantity': round(pred_qty, 2),
                'target_quantity': target_qty
            })

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)