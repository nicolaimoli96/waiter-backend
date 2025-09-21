# app.py
# Flask backend with preflight-safe CORS for multiple frontends
# Includes /api/waiters, /api/recommend-categories, /api/simulate-daily, and login example

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

# CORS configuration: allow your frontends, common methods, and headers
CORS(
    app,
    origins=[
        "http://localhost:3000",
        "https://waiter-frontend.netlify.app",
        "https://waiter-goals-black.netlify.app",
        "https://waiter-company-goals.netlify.app"
    ],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

# Load the trained model, encoder, and categories
model = joblib.load('category_model.joblib')
enc = joblib.load('encoder.joblib')
categories = joblib.load('categories.pkl')

# Load waiter names (Clerk Name) from CSV at startup
df = pd.read_csv('FM_training_data.csv')
waiters = sorted(df['Clerk Name'].unique())

# ---------------------
# Example login endpoint
# ---------------------
@app.route('/api/auth/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200  # Respond OK to preflight

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Replace this with your actual authentication logic
    if username == "admin" and password == "admin":
        return jsonify({"success": True})
    else:
        return jsonify({"success": False}), 401

# ---------------------
# Get waiter names
# ---------------------
@app.route('/api/waiters', methods=['GET', 'OPTIONS'])
def get_waiters():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({'waiters': waiters})

# ---------------------
# Simulate daily
# ---------------------
@app.route('/api/simulate-daily', methods=['POST', 'OPTIONS'])
def simulate_daily():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json()
        day_of_week = data.get('day_of_week')
        weather = data.get('weather')
        daily_target = data.get('daily_target', 0)
        sales_done_today = data.get('sales_done_today', 0)

        # Validate
        if not all([day_of_week is not None, weather in ['rain', 'cloud', 'wind', 'sunny'], daily_target > 0]):
            return jsonify({'error': 'Invalid input'}), 400

        # Feature DataFrame
        current_features = pd.DataFrame({
            'day_of_week': [day_of_week],
            'weather_rain': [1 if weather == 'rain' else 0],
            'weather_cloud': [1 if weather == 'cloud' else 0],
            'weather_wind': [1 if weather == 'wind' else 0],
            'weather_sunny': [1 if weather == 'sunny' else 0]
        })

        # Add your existing logic here...
        # For now, return dummy response
        return jsonify({'message': 'Simulation processed'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------------
# Recommend categories
# ---------------------
@app.route('/api/recommend-categories', methods=['POST', 'OPTIONS'])
def recommend_categories():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json()
        day = data.get('day')
        session = data.get('session')
        weather = data.get('weather')
        waiter = data.get('waiter')

        if session == 'Lunch':
            session = 'Before5pm'
        elif session == 'Dinner':
            session = 'After5pm'

        if not all([day, session, weather, waiter]):
            return jsonify({'error': 'Missing or invalid input'}), 400

        input_df = pd.DataFrame({
            'Weekday': [day],
            'Session': [session],
            'Weather': [weather],
            'Clerk Name': [waiter]
        })

        X_input = enc.transform(input_df)
        preds = model.predict(X_input)[0]

        cat_preds = {cat: preds[i] for i, cat in enumerate(categories)}
        sorted_cats = sorted(cat_preds.items(), key=lambda x: x[1], reverse=True)[:3]

        recommendations = []
        for cat, pred_qty in sorted_cats:
            target_qty = int(round(pred_qty * 1.2))
            recommendations.append({
                'category': cat,
                'predicted_quantity': round(pred_qty, 2),
                'target_quantity': target_qty
            })

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------------
# Main
# ---------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
