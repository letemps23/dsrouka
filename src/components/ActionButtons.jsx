import React from 'react';
import axios from 'axios';
import './ActionButtons.css';

function ActionButtons({ onRefresh }) {
  const callEndpoint = async (endpoint) => {
    const res = await axios.get(`${process.env.REACT_APP_API_URL}/${endpoint}`);
    alert(res.data.message || "Opération terminée !");

    // Rafraîchir les graphiques après traitement
    if (endpoint === "entrainer" && onRefresh) {
      onRefresh();  // Signal au parent (App.jsx)
    }
  };

  return (
    <div className="action-buttons">
      <button onClick={() => callEndpoint("interpoler")}>📊 Interpoler les données</button>
      <button onClick={() => callEndpoint("entrainer")}>🧠 Entraîner LSTM</button>
      <button onClick={() => callEndpoint('api/arima-json', 'ARIMA exécuté et prédictions prêtes')}>📈 Entraîner ARIMA</button>
      <button onClick={() => callEndpoint('api/prophet-json', 'Prédiction Prophet terminée')}>🔮 Lancer Prophet</button>

    </div>
  );
}

export default ActionButtons;

