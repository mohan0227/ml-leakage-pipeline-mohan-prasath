import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

d = {
    "area_sqft": np.random.randint(500, 2000, 50),
    "num_bedrooms": np.random.randint(1, 5, 50),
    "age_years": np.random.randint(0, 10, 50),
}

df = pd.DataFrame(d)

print("TASK 1")
df["price_lakhs"] = (
    df["area_sqft"] * 0.08 + df["num_bedrooms"] * 10 - df["age_years"] * 1.5 + np.random.normal(0, 10, 50)
).round(2)

X = df[['area_sqft', 'num_bedrooms', 'age_years']]
y = df['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

for feature, coeff in zip(X_train.columns, model.coef_):
    print(f"{feature:15s}: {coeff:+.3f}")

print(f"{'intercept':15s}: {model.intercept_:+.3f}")

pred = model.predict(X_test)

result = pd.DataFrame({
    'Actual': y_test,
    'Predicted': pred,
})

print()
print(result.head())

print("\nTASK 2")
mae = mean_absolute_error(result['Actual'], result['Predicted'])
rmse = np.sqrt(mean_squared_error(result['Actual'], result['Predicted']))
r2 = r2_score(result['Actual'], result['Predicted'])

print(f"{'MAE':12s}: {mae:.2f} lakhs")
print(f"{'RMSE':12s}: {rmse:.2f} lakhs")
print(f"{'R-squared':12s}: {r2:.3f}")


print("\nTask 3")
residual = result['Actual'] - result['Predicted']

plt.hist(residual, bins=10)
plt.title("Residual Plot")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show(block=False)
plt.pause(3)
plt.close()