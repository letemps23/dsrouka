import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

def entrainer_modele_lstm(df):
    df['HOUR'] = df['DATETIME'].dt.hour
    df['DAY_OF_WEEK'] = df['DATETIME'].dt.dayofweek
    df['WEEK'] = df['DATETIME'].dt.isocalendar().week

    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f"LAG_{lag}"] = df["CONSOMMATION_TOTALE"].shift(lag)

    df["MA_6"] = df["CONSOMMATION_TOTALE"].rolling(6).mean()
    df["MA_12"] = df["CONSOMMATION_TOTALE"].rolling(12).mean()

    df = df.dropna().reset_index(drop=True)

    features = ['HOUR', 'DAY_OF_WEEK', 'WEEK'] + \
               [f"LAG_{lag}" for lag in [1, 2, 3, 6, 12, 24, 48]] + \
               ["MA_6", "MA_12"]
    target = "CONSOMMATION_TOTALE"

    X = df[features].values
    y = df[target].values
    datetimes = df["DATETIME"].values
    df.to_excel("Dataset_Pret_Prediction_LSTM.xlsx", index=False)

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_inv = scaler_y.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
    mae = mean_absolute_error(y_test_inv, y_pred)
    mape = np.mean(np.abs((y_test_inv - y_pred) / y_test_inv)) * 100

    result_df = pd.DataFrame({
        "DATETIME": datetimes[-len(y_test):],
        "Valeur Réelle": y_test_inv.flatten(),
        "Prédiction LSTM": y_pred.flatten()
    })
    total_reel = np.sum(y_test_inv)
    total_pred = np.sum(y_pred)

    result_df["Total Prédiction"] = total_pred
    result_df["Total Réel"] = total_reel
    result_df.to_excel("Resultats_Prediction_LSTM.xlsx", index=False)



    return result_df, {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 2)}
