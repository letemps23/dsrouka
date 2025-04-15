from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from preprocessing import charger_et_nettoyer, interpoler_serie, analyse_avancee
from model_lstm import entrainer_modele_lstm
from fastapi.responses import FileResponse
import os
from fastapi import Query
import re
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import unicodedata
import matplotlib.pyplot as plt
import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
import joblib
from fastapi import APIRouter
from datetime import timedelta
from tensorflow.keras.models import load_model
#matplotlib.use('Agg')


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
    

# chargement du fichier

@app.get("/api/charger-donnees")
def charger_donnees_excel_flexibles():
    dossier = "data/"
    fichiers_excel = [f for f in os.listdir(dossier) if f.endswith(".xlsx")]

    mois_map = {
        "janvier": "JANVIER", "fevrier": "FEVRIER", "février": "FEVRIER",
        "mars": "MARS", "avril": "AVRIL", "mai": "MAI", "juin": "JUIN",
        "juillet": "JUILLET", "aout": "AOUT", "août": "AOUT",
        "septembre": "SEPTEMBRE", "octobre": "OCTOBRE",
        "novembre": "NOVEMBRE", "decembre": "DECEMBRE", "décembre": "DECEMBRE"
    }

    colonnes_utiles = ['DATE', 'HEURES', 'LOME', 'ANFOIN', 'ATAKPAME', 'KARA',
                       'SULZER1', 'SULZER2', 'CTL', 'KPIME', 'KARA_PROD']

    def normaliser(texte):
        return unicodedata.normalize('NFKD', texte).encode('ascii', 'ignore').decode('utf-8').lower()

    toutes_les_donnees = []
    fichiers_utilises = []
    fichiers_ignores = []

    for fichier in fichiers_excel:
        chemin = os.path.join(dossier, fichier)
        nom_normalise = normaliser(fichier)

        match = re.search(r"(janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|septembre|octobre|novembre|decembre|décembre)[ _-]+(\d{4})", nom_normalise)

        if match:
            mois_nom = match.group(1)
            annee = int(match.group(2))
            feuille = mois_map.get(mois_nom)

            if feuille:
                try:
                    df = pd.read_excel(chemin, sheet_name=feuille, usecols=colonnes_utiles)
                    df = df.fillna(0)

                    conso_cols = ['LOME', 'ANFOIN', 'ATAKPAME', 'KARA', 'SULZER1',
                                  'SULZER2', 'CTL', 'KPIME', 'KARA_PROD']
                    df["CONSOMMATION_TOTALE"] = df[conso_cols].sum(axis=1)

                    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
                    df["HEURES"] = df["HEURES"].astype(str).str.strip().str.replace("H", ":", regex=False)
                    df["HEURES"] = df["HEURES"].apply(lambda x: x + ":00" if len(x) <= 5 else x)

                    df["DATETIME"] = pd.to_datetime(
                        df["DATE"].dt.strftime("%Y-%m-%d") + " " + df["HEURES"],
                        format="%Y-%m-%d %H:%M:%S",
                        errors="coerce"
                    )

                    df = df.dropna(subset=["DATETIME"])
                    df = df.drop(columns=["DATE", "HEURES"])

                    df["ANNEE"] = annee
                    df["MOIS"] = feuille

                    toutes_les_donnees.append(df)
                    fichiers_utilises.append(fichier)

                except Exception as e:
                    fichiers_ignores.append((fichier, str(e)))
            else:
                fichiers_ignores.append((fichier, "Mois non reconnu"))
        else:
            fichiers_ignores.append((fichier, "Format nom de fichier invalide"))

    if not toutes_les_donnees:
        return JSONResponse(content={
            "error": "Aucun fichier valide traité.",
            "fichiers_ignores": fichiers_ignores
        }, status_code=404)

    # ✅ Fusionner les données
    df_final = pd.concat(toutes_les_donnees, ignore_index=True)

    # ✅ Nettoyage et filtrage datetime
    df_final = df_final.dropna(subset=["DATETIME"])
    df_final["DATETIME"] = pd.to_datetime(df_final["DATETIME"], errors="coerce")
    df_final = df_final[(df_final["DATETIME"] >= "2014-01-01") & (df_final["DATETIME"] <= "2019-12-31")]

    # ✅ Enregistrement du dataset nettoyé
    df_final.to_excel("Dataset_6_ans_corrige.xlsx", index=False)

    return JSONResponse(content={
        "message": "✅ Données fusionnées et nettoyées avec succès.",
        "lignes_totales": len(df_final),
        "fichier_sortie": "Dataset_6_ans_corrige.xlsx",
        "fichiers_utilises": fichiers_utilises,
        "fichiers_ignores": fichiers_ignores
    })




#graphique


#from fastapi.responses import StreamingResponse, JSONResponse
#import pandas as pd
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import io

@app.get("/api/graph-6ans")
def graphique_6_ans():
    try:
        # Charger les données depuis le fichier propre
        df = pd.read_excel("Dataset_6_ans_corrige.xlsx", parse_dates=["DATETIME"])

        # Vérification de la colonne
        if "DATETIME" not in df.columns or "CONSOMMATION_TOTALE" not in df.columns:
            return {"error": "Les colonnes nécessaires sont manquantes dans le fichier."}

        # Tri par datetime au cas où
        df = df.sort_values(by="DATETIME")

        # Tracer le graphique
        plt.figure(figsize=(14, 6))
        plt.plot(df["DATETIME"], df["CONSOMMATION_TOTALE"], label="Consommation totale")
        plt.title("📊 Évolution de la consommation totale (2014 - 2019)")
        plt.xlabel("Date et heure")
        plt.ylabel("Consommation (MW)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Sauvegarde du graphique en mémoire
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        return {"error": f"Erreur lors de la génération du graphique : {str(e)}"}
    

   #from fastapi.responses import JSONResponse

@app.get("/api/consommation-json")
def consommation_json(
    annee: int = Query(..., ge=2014, le=2019),
    mois: int = Query(None, ge=1, le=12)
):
    fichier = "Dataset_6_ans_corrige.xlsx"
    if not os.path.exists(fichier):
        return JSONResponse(status_code=404, content={"error": "Fichier introuvable"})

    try:
        df = pd.read_excel(fichier, parse_dates=["DATETIME"])
        df = df.dropna(subset=["DATETIME", "CONSOMMATION_TOTALE"])
        df["ANNEE"] = df["DATETIME"].dt.year
        df["MOIS"] = df["DATETIME"].dt.month

        # 🔍 Filtrage par année et mois
        df = df[df["ANNEE"] == annee]
        if mois:
            df = df[df["MOIS"] == mois]

        # ❌ Supprimer les lignes où consommation = 0
        df = df[df["CONSOMMATION_TOTALE"] > 0]

        # ✅ Format JSON compatible avec Chart.js
        df["DATETIME"] = df["DATETIME"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return JSONResponse(content=df[["DATETIME", "CONSOMMATION_TOTALE"]].to_dict(orient="records"))

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    





@app.get("/api/pretraitement-cnn-lstm")
def pretraitement_cnn_lstm():
    try:
        df = pd.read_excel("Dataset_6_ans_corrige.xlsx", parse_dates=["DATETIME"])
        df = df[["DATETIME", "CONSOMMATION_TOTALE"]].dropna()

        # Normalisation
        scaler = MinMaxScaler()
        df["CONSO_NORM"] = scaler.fit_transform(df[["CONSOMMATION_TOTALE"]])

        # Séquences
        window_size = 24
        X, y = [], []
        for i in range(len(df) - window_size):
            seq = df["CONSO_NORM"].values[i:i+window_size]
            target = df["CONSO_NORM"].values[i+window_size]
            X.append(seq)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        # Reshape pour CNN-LSTM
        X = X.reshape((X.shape[0], 4, 6, 1))  # 4 sous-séquences de 6 pas

        # Sauvegarde
        np.save("X_cnn_lstm.npy", X)
        np.save("y_cnn_lstm.npy", y)
        joblib.dump(scaler, "scaler_cnn_lstm.pkl")

        return JSONResponse(content={
            "message": "✅ Données préparées avec succès pour CNN-LSTM.",
            "shape_X": X.shape,
            "shape_y": y.shape
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})





@app.get("/api/entrainement-cnn-lstm")
def entrainer_cnn_lstm():
    try:
        X = np.load("X_cnn_lstm.npy")
        y = np.load("y_cnn_lstm.npy")

        model = Sequential([
            TimeDistributed(Conv1D(64, 3, activation='relu'), input_shape=(4, 6, 1)),
            TimeDistributed(MaxPooling1D(2)),
            TimeDistributed(Flatten()),
            LSTM(50, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        history = model.fit(X, y, epochs=20, batch_size=32,
                            validation_split=0.2,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                            verbose=1)

        model.save("cnn_lstm_model.h5")

        return JSONResponse(content={
            "message": "✅ Entraînement CNN-LSTM terminé.",
            "epochs": len(history.history['loss']),
            "val_loss_finale": history.history['val_loss'][-1]
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.get("/api/prediction-cnn-lstm-multi")
def prediction_multi_horizon():
    try:
        df = pd.read_excel("Dataset_6_ans_corrige.xlsx", parse_dates=["DATETIME"])
        df = df[["DATETIME", "CONSOMMATION_TOTALE"]].dropna()

        # Préparation
        scaler = joblib.load("scaler_cnn_lstm.pkl")
        df["CONSO_NORM"] = scaler.transform(df[["CONSOMMATION_TOTALE"]])

        window = df["CONSO_NORM"].values[-24:]  # Derniers 24 points
        if len(window) < 24:
            raise ValueError("Pas assez de données.")

        #model = load_model("cnn_lstm_model.h5")
        model = load_model("cnn_lstm_model.h5", compile=False)

        # Prédiction multi-horizon
        horizons = {
            "jour": 48,
            "semaine": 336,
            #"mois": 1440,
            #"annee": 17520
        }

        predictions = {}
        now = df["DATETIME"].max()

        for label, steps in horizons.items():
            seq = window.copy()
            pred_norms = []

            for _ in range(steps):
                x = seq.reshape((1, 4, 6, 1))
                y = model.predict(x, verbose=0)[0][0]
                pred_norms.append(y)
                seq = np.append(seq[1:], y)

            y_pred = scaler.inverse_transform(np.array(pred_norms).reshape(-1, 1)).flatten()
            timestamps = [now + timedelta(minutes=30 * (i + 1)) for i in range(steps)]

            predictions[label] = {
                "datetime": [ts.isoformat() for ts in timestamps],
                "valeurs": y_pred.tolist()
            }

        return {
            "message": "✅ Prédictions multiples générées.",
            "predictions": predictions
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})