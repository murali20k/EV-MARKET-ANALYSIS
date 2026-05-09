
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("ev_market_2026.csv")

print(df.head())

# Dataset Info
print(df.info())

# Missing Values
print(df.isnull().sum())

# Fill missing values
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# =========================
# Exploratory Data Analysis
# =========================

# Top Brands
plt.figure(figsize=(10,5))
df['brand'].value_counts().head(10).plot(kind='bar')
plt.title("Top EV Brands")
plt.xlabel("Brand")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Battery vs Range
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x='battery_capacity_kwh',
    y='range_miles',
    hue='brand'
)
plt.title("Battery Capacity vs Range")
plt.tight_layout()
plt.show()

# =========================
# Machine Learning
# =========================

features = [
    'battery_capacity_kwh',
    'range_miles',
    'charging_speed_kw',
    'horsepower',
    'torque_nm',
    'weight_kg'
]

X = df[features]
y = df['price_usd']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# Feature Importance
importance = pd.Series(model.coef_, index=features)
print("\nFeature Importance:")
print(importance.sort_values(ascending=False))

# Sample Prediction
new_data = pd.DataFrame({
    'battery_capacity_kwh':[80],
    'range_miles':[350],
    'charging_speed_kw':[150],
    'horsepower':[400],
    'torque_nm':[600],
    'weight_kg':[2200]
})

prediction = model.predict(new_data)

print("\nPredicted EV Price:", prediction[0])
