# 1. Import Library
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from xgboost import train
from google.colab import files
from xgboost import XGBRegressor, callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve

# 2. Memuat Dataset 
uploaded = files.upload()

# Dapatkan nama file yang diupload
file_name = list(uploaded.keys())[0]

# Cek ekstensi file
if file_name.endswith('.csv'):
    df = pd.read_csv(file_name)
elif file_name.endswith('.xlsx'):
    df = pd.read_excel(file_name)
else:
    raise ValueError("Format file tidak didukung. Harap upload file CSV atau XLSX.")

# Lihat struktur dataset
print("\nStruktur Dataset:")
print(df.head())

# 3. Preprocessing Data
# Pastikan tidak ada nilai null
print("\nJumlah nilai null:\n", df.isnull().sum())

# Jika ada nilai null, Anda bisa menghapus baris atau mengisi dengan nilai rata-rata/mode
df = df.dropna()  

# Pisahkan fitur (X) dan target (y)
X = df.drop(columns=['FloodProbability'])  
y = df['FloodProbability']  

# Encode categorical data jika ada 
# Contoh: X = pd.get_dummies(X)

# Scale data menggunakan MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 4. Membagi Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# 5. Hyperparameter Tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [200, 300, 700],
    'max_depth': [2, 3, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Cross-validation dengan GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Menampilkan parameter terbaik
print("\nBest Parameters:", grid_search.best_params_)

# Model dengan parameter terbaik
best_model = grid_search.best_estimator_

# 6. Early Stopping dengan Callback
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Definisikan callback untuk early stopping
early_stopping = callback.EarlyStopping(
    rounds=10,
    save_best=True,
    maximize=False
)

# Latih model dengan early stopping
eval_set = [(X_val, y_val)]
best_model = train(
    params=best_model.get_xgb_params(),
    dtrain=xgb.DMatrix(X_train_final, label=y_train_final),
    num_boost_round=best_model.get_params()['n_estimators'],
    evals=[(xgb.DMatrix(X_val, label=y_val), 'validation')],
    early_stopping_rounds=10,
    verbose_eval=False
)

# 7. Evaluasi Model
# Prediksi pada data testing
y_pred = best_model.predict(xgb.DMatrix(X_test))

# Hitung metrik evaluasi
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluasi Model:")
print(f"Mean Squared Error (MSE): {mse:.7f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.7f}")
print(f"Mean Absolute Error (MAE): {mae:.7f}")
print(f"R-squared (R2): {r2:.7f}")

# 8. Visualisasi Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    grid_search.best_estimator_,
    X_train, y_train, cv=3, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Hitung mean dan std dari skor training dan validasi
train_scores_mean = -train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
val_scores_mean = -val_scores.mean(axis=1)
val_scores_std = val_scores.std(axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation Score")
plt.xlabel("Training Examples")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.show()

# 9. Visualisasi Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label="Prediksi")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Fit")
plt.xlabel("Actual Flood Probability")
plt.ylabel("Predicted Flood Probability")
plt.title("Actual vs Predicted Flood Probability")
plt.legend()
plt.grid(True)
plt.show()

# Tambahkan evaluasi error pada setiap data point
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, color='orange', edgecolor='black')
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Errors")
plt.grid(True)
plt.show()
