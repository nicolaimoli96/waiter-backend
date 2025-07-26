# train_model.py
# Run this script first to train and save the model, encoder, and categories list.
# Make sure sales_data.csv is in the same directory.

import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load the data
df = pd.read_csv('sales_data.csv')

# Get unique categories (sorted for consistency)
categories = sorted(df['Category'].unique())

# Compute the most common weather per Date, Session, Waiter
weather_mode = df.groupby(['Date', 'Session', 'Waiter'])['Weather'].agg(
    lambda x: x.value_counts().idxmax() if not x.empty else None
).reset_index(name='Weather')

# Aggregate quantities: sum Quantity per Date, Session, Waiter, Category
qty_group = df.groupby(['Date', 'Session', 'Waiter', 'Category'])['Quantity'].sum().reset_index()

# Pivot to wide format: one column per category
pivot_qty = qty_group.pivot(
    index=['Date', 'Session', 'Waiter'],
    columns='Category',
    values='Quantity'
).fillna(0).reset_index()

# Merge with weather_mode
data = pivot_qty.merge(weather_mode, on=['Date', 'Session', 'Waiter'])

# Get Day per Date (assuming consistent per Date)
day_group = df[['Date', 'Day']].drop_duplicates()
data = data.merge(day_group, on='Date')

# Features and targets
feature_cols = ['Day', 'Session', 'Weather', 'Waiter']
X_df = data[feature_cols]
y = data[categories]  # Targets: quantity per category

# One-hot encode categorical features
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X = enc.fit_transform(X_df)

# Train RandomForestRegressor (multi-output by default for 2D targets)
# RandomForest is chosen as it's robust for this tabular data with categorical features,
# handles multi-output regression well, and doesn't require scaling.
# For "best" model, you could experiment with XGBoostRegressor or GradientBoostingRegressor,
# but RF is a strong baseline. Use cross-validation in production to tune.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model, encoder, and categories
joblib.dump(model, 'category_model.joblib')
joblib.dump(enc, 'encoder.joblib')
joblib.dump(categories, 'categories.pkl')

print("Model trained and saved successfully.")