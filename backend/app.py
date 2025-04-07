from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from preprocessing import charger_et_nettoyer, interpoler_serie, analyse_avancee
from model_lstm import entrainer_modele_lstm
from fastapi.responses import FileResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error



#from fastapi import APIRouter
#from fastapi.responses import JSONResponse
#import pandas as pd
import statsmodels.api as sm

app = FastAPI(
    title="API LSTM Consommation Énergie",
    description="API de traitement complet et prédiction avec LSTM",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATAFRAME = {}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    with open("temp.xlsx", "wb") as f:
        f.write(content)
    df = charger_et_nettoyer("temp.xlsx")
    DATAFRAME["clean"] = df
    df.to_excel("Consommation_Filtrée_Avril_2016.xlsx", index=False)
    return {"message": f"{len(df)} lignes chargées et nettoyées."}

@app.get("/interpoler/")
def interpoler():
    if "clean" not in DATAFRAME:
        return {"error": "Aucune donnée chargée."}
    df_interp = interpoler_serie(DATAFRAME["clean"])
    DATAFRAME["interpolée"] = df_interp
    df_interp.to_excel("Dataset_Corrige_Interpolé.xlsx", index=False)
    import matplotlib.pyplot as plt

    # Tracer et enregistrer la courbe d’évolution interpolée
    def tracer_courbe_evolution(df_interp):
        plt.figure(figsize=(14, 6))
        plt.plot(df_interp["DATETIME"], df_interp["CONSOMMATION_TOTALE"], color='blue', linewidth=2)
        plt.title("Évolution de la consommation totale (interpolée à 30 minutes)")
        plt.xlabel("Date et Heure")
        plt.ylabel("Consommation (MW)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("Courbe_Evolution_Consommation_30min.png")
        plt.close()

    analyse_avancee(df_interp)
    tracer_courbe_evolution(df_interp)
    return {"message": "Interpolation à 30min réalisée.", "total_points": len(df_interp)}

@app.get("/entrainer/")
def entrainer():
    if "interpolée" not in DATAFRAME:
        return {"error": "Aucune série interpolée disponible."}
    df_pred, metrics = entrainer_modele_lstm(DATAFRAME["interpolée"])
    df_pred.to_excel("Predictions_LSTM_API.xlsx", index=False)
    return {
        "message": "Modèle entraîné et prédictions générées.",
        "metrics": metrics
    }

@app.get("/")
def root():
    return {"message": "API LSTM opérationnelle. Utilisez /upload/, /interpoler/, /entrainer/"}

# 📥 Fichier : Données filtrées (après nettoyage)
@app.get("/telecharger/consommation-filtree")
def telecharger_conso_filtree():
    path = "Consommation_Filtrée_Avril_2016.xlsx"
    if os.path.exists(path):
        return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="Consommation_Filtrée_Avril_2016.xlsx")
    return {"error": "Fichier non trouvé"}

# 📥 Fichier : Données interpolées à 30 min
@app.get("/telecharger/dataset-corrige")
def telecharger_dataset_corrige():
    path = "Dataset_Corrige_Interpolé.xlsx"
    if os.path.exists(path):
        return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="Dataset_Corrige_Interpolé.xlsx")
    return {"error": "Fichier non trouvé"}

# 📥 Fichier : Dataset prêt pour la prédiction
@app.get("/telecharger/dataset-prediction")
def telecharger_dataset_prediction():
    path = "Dataset_Pret_Prediction_LSTM.xlsx"
    if os.path.exists(path):
        return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="Dataset_Pret_Prediction_LSTM.xlsx")
    return {"error": "Fichier non trouvé"}

# 📥 Fichier : Courbe PNG de la prédiction
@app.get("/telecharger/courbe")
def telecharger_courbe():
    path = "Courbe_Prediction_LSTM.png"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", filename="Courbe_Prediction_LSTM.png")
    return {"error": "Fichier non trouvé"}

# 📥 Fichier : Décomposition de la tendance (PNG)
@app.get("/telecharger/decomposition")
def telecharger_decomposition():
    path = "Decomposition_Serie_Temporelle.png"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", filename="Decomposition_Serie_Temporelle.png")
    return {"error": "Fichier non trouvé"}

# 📥 Fichier : Analyse des pics et creux
@app.get("/telecharger/pics-creux")
def telecharger_pics_creux():
    path = "Pics_Creux_Journaliers.xlsx"
    if os.path.exists(path):
        return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="Pics_Creux_Journaliers.xlsx")
    return {"error": "Fichier non trouvé"}

# 📥 Fichier : Résultats de prédiction avec totaux
@app.get("/telecharger/resultats-prediction")
def telecharger_resultats_prediction():
    path = "Resultats_Prediction_LSTM.xlsx"
    if os.path.exists(path):
        return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="Resultats_Prediction_LSTM.xlsx")
    return {"error": "Fichier non trouvé"}

# 📥 Fichier : Courbe d'évolution à 30 min
@app.get("/telecharger/courbe-evolution-30min")
def telecharger_courbe_evolution():
    path = "Courbe_Evolution_Consommation_30min.png"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", filename="Courbe_Evolution_Consommation_30min.png")
    return {"error": "Fichier non trouvé"}


@app.get("/api/interpolation-json")
def interpolation_json():
    path = "Dataset_Corrige_Interpolé.xlsx"
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Fichier manquant"}, status_code=404)

    df = pd.read_excel(path)

    # Convertir les timestamps en string lisibles
    df["DATETIME"] = df["DATETIME"].astype(str)

    df = df.replace([np.nan, np.inf, -np.inf], 0)
    df = df[["DATETIME", "CONSOMMATION_TOTALE"]]

    return JSONResponse(content=df.to_dict(orient="records"))


@app.get("/api/prediction-json")
def prediction_json():
    path = "Resultats_Prediction_LSTM.xlsx"
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Fichier manquant"}, status_code=404)

    df = pd.read_excel(path)

    # Convertir les timestamps en string lisibles
    df["DATETIME"] = df["DATETIME"].astype(str)

    df = df.replace([np.nan, np.inf, -np.inf], 0)
    df = df[["DATETIME", "Valeur Réelle", "Prédiction LSTM"]]

    return JSONResponse(content=df.to_dict(orient="records"))


# Route pour la décomposition de la série temporelle
@app.get("/api/decomposition-json")
def decomposition_json():
    import numpy as np
    file_path = "Dataset_Pret_Prediction_LSTM.xlsx"
    df = pd.read_excel(file_path)

    if "DATETIME" not in df.columns or "CONSOMMATION_TOTALE" not in df.columns:
        return JSONResponse(content={"error": "Colonnes manquantes"}, status_code=400)

    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df.set_index("DATETIME", inplace=True)
    df = df.asfreq("30min")

    serie = df["CONSOMMATION_TOTALE"].dropna()
    decomposition = sm.tsa.seasonal_decompose(serie, model="additive", period=48)

    result = {
        "datetime": serie.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "trend": decomposition.trend.replace({np.nan: None}).tolist(),
        "seasonal": decomposition.seasonal.replace({np.nan: None}).tolist(),
        "resid": decomposition.resid.replace({np.nan: None}).tolist()
    }
    return JSONResponse(content=result)




# Route pour les pics et creux journaliers
@app.get("/api/pics-creux-json")
def pics_creux_json():
    file_path = "Dataset_Pret_Prediction_LSTM.xlsx"
    df = pd.read_excel(file_path)
    
    df = df.dropna(subset=["DATETIME", "CONSOMMATION_TOTALE"])
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["DATE"] = df["DATETIME"].dt.date

    # Groupement par jour
    daily = df.groupby("DATE")["CONSOMMATION_TOTALE"]
    result = {
        "date": daily.max().index.astype(str).tolist(),
        "pics": daily.max().tolist(),
        "creux": daily.min().tolist()
    }
    return JSONResponse(content=result)

#ARIMA
@app.get("/api/arima-json")
def arima_json():
    file_path = "Dataset_Corrige_Interpolé.xlsx"
    df = pd.read_excel(file_path)

    # Assurer les colonnes nécessaires
    if "DATETIME" not in df.columns or "CONSOMMATION_TOTALE" not in df.columns:
        return JSONResponse(content={"error": "Colonnes manquantes"}, status_code=400)

    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df = df.set_index("DATETIME")
    serie = df["CONSOMMATION_TOTALE"].dropna()

    # Split train/test
    split = int(len(serie) * 0.8)
    train, test = serie[:split], serie[split:]

    model = ARIMA(train, order=(5, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    forecast.index = test.index

    # Nettoyage des NaN
    combined = pd.concat([test, forecast], axis=1)
    combined.columns = ["Réel", "Prévu"]
    combined = combined.dropna()

    # JSON prêt pour React
    result = {
        "datetime": combined.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "reel": combined["Réel"].tolist(),
        "prevision": combined["Prévu"].tolist()
    }

    return JSONResponse(content=result)

#PROPHET
@app.get("/api/prophet-json")
def prophet_json():
    try:
        df = pd.read_excel("Dataset_Corrige_Interpolé.xlsx")
        if "DATETIME" not in df.columns or "CONSOMMATION_TOTALE" not in df.columns:
            return JSONResponse(content={"error": "Colonnes manquantes"}, status_code=400)

        df["DATETIME"] = pd.to_datetime(df["DATETIME"])
        df = df.dropna(subset=["DATETIME", "CONSOMMATION_TOTALE"])

        df_prophet = df.rename(columns={"DATETIME": "ds", "CONSOMMATION_TOTALE": "y"})

        split = int(len(df_prophet) * 0.8)
        train = df_prophet.iloc[:split]
        test = df_prophet.iloc[split:]

        model = Prophet()
        model.fit(train)

        future = model.make_future_dataframe(periods=len(test), freq="30min")
        forecast = model.predict(future)

        # Filtrer les dates de test uniquement
        forecast_filtered = forecast[forecast["ds"].isin(test["ds"])]
        # Calcul des scores
        rmse = round(np.sqrt(mean_squared_error(test["y"], forecast_filtered["yhat"])), 2)
        mae = round(mean_absolute_error(test["y"], forecast_filtered["yhat"]), 2)
        mape = round(np.mean(np.abs((test["y"] - forecast_filtered["yhat"]) / test["y"])) * 100, 2)

        result = {
            "datetime": forecast_filtered["ds"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "reel": test["y"].tolist(),
            "prevision": forecast_filtered["yhat"].tolist(),
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)