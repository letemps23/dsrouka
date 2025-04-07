import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Charger les données
file_path = "backend\Dataset_Corrige_Interpolé.xlsx"
df = pd.read_excel(file_path)
df["DATETIME"] = pd.to_datetime(df["DATETIME"])
df = df.set_index("DATETIME")
serie = df["CONSOMMATION_TOTALE"].dropna()

# Séparer en train/test (80% / 20%)
split_index = int(len(serie) * 0.8)
train, test = serie[:split_index], serie[split_index:]

# Entraînement du modèle ARIMA
model = ARIMA(train, order=(5, 1, 2))
model_fit = model.fit()

# Prédiction
forecast = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast, index=test.index)

# Évaluation
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test - forecast) / test)) * 100

print("📊 Résultats ARIMA :")
print(f"✅ RMSE : {rmse:.2f}")
print(f"✅ MAE  : {mae:.2f}")
print(f"✅ MAPE : {mape:.2f}%")

# Sauvegarde des résultats
result_df = pd.DataFrame({
    "DATETIME": test.index,
    "Réel": test.values,
    "Prévu_ARIMA": forecast.values
})
result_df.to_excel("Predictions_ARIMA.xlsx", index=False)

# Courbe
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label="Valeurs Réelles", color='blue')
plt.plot(forecast.index, forecast, label="Prévisions ARIMA", color='red', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Consommation (MW)")
plt.title("ARIMA : Prédictions vs Réel")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Courbe_ARIMA.png")
plt.show()
