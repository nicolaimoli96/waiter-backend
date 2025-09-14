# train_model.py
# Retrained to predict average daily/session category quantities under given conditions (Weekday, Session, Weather, Clerk Name).
# Aggregates sums per category (Sales Mix Group 2) across historical dates with same conditions, then averages.
# Run this to save new model/artifacts. Uses new dataset 'FM_training_data.csv'.

import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load the data
df = pd.read_csv('FM_training_data.csv')

# Get unique categories from 'Sales Mix Group 2' (sorted for consistency)
categories = sorted(df['Sales Mix Group 2'].unique())

# Group by 'Weekday', 'Session', 'Weather', 'Clerk Name', 'Sales Mix Group 2' to sum 'Quantity (Sum)' across all matching dates
qty_group = df.groupby(['Weekday', 'Session', 'Weather', 'Clerk Name', 'Sales Mix Group 2'])['Quantity (Sum)'].sum().reset_index()

# Count number of unique 'Business Date' per group to compute average daily/session qty = sum / date_count
date_counts = df.groupby(['Weekday', 'Session', 'Weather', 'Clerk Name'])['Business Date'].nunique().reset_index(name='date_count')

# Merge and compute average
qty_group = qty_group.merge(date_counts, on=['Weekday', 'Session', 'Weather', 'Clerk Name'])
qty_group['Average Quantity'] = qty_group['Quantity (Sum)'] / qty_group['date_count']

# Pivot averages to wide: one column per category
pivot_qty = qty_group.pivot(
    index=['Weekday', 'Session', 'Weather', 'Clerk Name'],
    columns='Sales Mix Group 2',
    values='Average Quantity'
).fillna(0).reset_index()

# Features and targets
feature_cols = ['Weekday', 'Session', 'Weather', 'Clerk Name']
X_df = pivot_qty[feature_cols]
y = pivot_qty[categories]  # Targets: average quantity per category

# One-hot encode categorical features
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X = enc.fit_transform(X_df)

# Train RandomForestRegressor (multi-output by default for 2D targets)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model, encoder, and categories
joblib.dump(model, 'category_model.joblib')
joblib.dump(enc, 'encoder.joblib')
joblib.dump(categories, 'categories.pkl')

print("Model trained and saved successfully.")