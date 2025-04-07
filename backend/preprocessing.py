import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import statsmodels.api as sm

def charger_et_nettoyer(fichier):
    df = pd.read_excel(fichier, sheet_name="AVRIL")

    colonnes_conso = ['LOME', 'ANFOIN', 'ATAKPAME', 'KARA', 'SULZER1',
                      'SULZER2', 'CTL', 'KPIME', 'KARA_PROD']
    df[colonnes_conso] = df[colonnes_conso].fillna(0)
    df["CONSOMMATION_TOTALE"] = df[colonnes_conso].sum(axis=1)

    df["HEURES_FORMATTED"] = df["HEURES"].astype(str).apply(lambda h: re.sub(r"(\d{2})H(\d{2})", r"\1:\2:00", h))
    df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')
    df["DATETIME"] = pd.to_datetime(df["DATE"].astype(str) + " " + df["HEURES_FORMATTED"], errors='coerce')
    df = df[['DATETIME'] + colonnes_conso + ['CONSOMMATION_TOTALE']].dropna()
    df = df[(df["DATETIME"] >= "2016-04-01") & (df["DATETIME"] <= "2016-04-30")]
    df = df.drop_duplicates(subset="DATETIME", keep="first")
    return df

def interpoler_serie(df):
    df = df.set_index("DATETIME").sort_index()
    df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='30min'))
    df.index.name = "DATETIME"
    df["CONSOMMATION_TOTALE"] = df["CONSOMMATION_TOTALE"].interpolate()
    return df.reset_index()

def analyse_avancee(df):
    df['DATE_ONLY'] = df['DATETIME'].dt.date
    daily_max = df.groupby('DATE_ONLY')["CONSOMMATION_TOTALE"].max()
    daily_min = df.groupby('DATE_ONLY')["CONSOMMATION_TOTALE"].min()

    df_pic_creux = pd.DataFrame({"PIC": daily_max, "CREUX": daily_min})
    df_pic_creux.to_excel("Pics_Creux_Journaliers.xlsx")

    # Décomposition
    df_ts = df.set_index("DATETIME")[["CONSOMMATION_TOTALE"]].copy()
    decomposition = sm.tsa.seasonal_decompose(df_ts, model="additive", period=48)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    decomposition.trend.plot(ax=axes[0], title="Tendance")
    decomposition.seasonal.plot(ax=axes[1], title="Saisonnalité")
    decomposition.resid.plot(ax=axes[2], title="Résidus")
    plt.tight_layout()
    plt.savefig("Decomposition_Serie_Temporelle.png")