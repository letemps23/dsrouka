import React, { useState } from 'react';
import axios from 'axios';
import './ActionButtons.css';

function ActionButtons({ onRefresh }) {
  const [metrics, setMetrics] = useState(null); // Pour afficher les scores

  const callEndpoint = async (endpoint, successMessage, saveMetrics = false) => {
    try {
      const res = await axios.get(`${process.env.REACT_APP_API_URL}/${endpoint}`);
      alert(successMessage || res.data.message || "✅ Opération réussie !");

      if (saveMetrics && res.data.metrics) {
        setMetrics(res.data.metrics);
      }

      if (onRefresh) onRefresh();
    } catch (err) {
      console.error(err);
      alert("❌ Une erreur est survenue.");
    }
  };

  return (
    <div className="action-buttons">
      <button onClick={() => callEndpoint("interpoler", "✅ Interpolation terminée")}>📊 Interpoler les données</button>
      <button onClick={() => callEndpoint("entrainer", "✅ Entraînement LSTM terminé")}>🧠 Entraîner LSTM</button>
      <button onClick={() => callEndpoint("api/arima-json", "✅ ARIMA exécuté")}>📈 Entraîner ARIMA</button>
      <button onClick={() => callEndpoint("api/prophet-json", "✅ Prophet terminé")}>🔮 Lancer Prophet</button>
      <button onClick={() => callEndpoint("api/pretraitement-cnn-lstm", "✅ Prétraitement CNN-LSTM terminé")}>⚙️ Prétraitement CNN-LSTM</button>
      <button onClick={() => callEndpoint("api/entrainement-cnn-lstm", "✅ Entraînement CNN-LSTM terminé", true)}>🤖 Entraîner CNN-LSTM</button>
      <button onClick={() => callEndpoint("api/prediction-cnn-lstm", "✅ Prédiction CNN-LSTM terminée")}>📈 Prédire CNN-LSTM</button>

      {metrics && (
        <div className="metrics-box">
          <h4>📊 Scores CNN-LSTM</h4>
          <p>RMSE : <strong>{metrics.rmse}</strong></p>
          <p>MAE  : <strong>{metrics.mae}</strong></p>
          <p>MAPE : <strong>{metrics.mape} %</strong></p>
        </div>
      )}
    </div>
  );
}

export default ActionButtons;
