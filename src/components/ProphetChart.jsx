import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';

function ProphetChart() {
  const [chartData, setChartData] = useState(null);
  const [scores, setScores] = useState({ rmse: null, mae: null, mape: null });

  useEffect(() => {
    axios.get(`${process.env.REACT_APP_API_URL}/api/prophet-json`)
      .then((response) => {
        const data = response.data;
        setChartData({
          labels: data.datetime,
          datasets: [
            {
              label: 'Valeurs Réelles',
              data: data.reel,
              borderColor: 'blue',
              fill: false,
            },
            {
              label: 'Prévisions Prophet',
              data: data.prevision,
              borderColor: 'green',
              borderDash: [5, 5],
              fill: false,
            },
          ],
        });
        setScores({
            rmse: data.rmse,
            mae: data.mae,
            mape: data.mape
          });
      })
      .catch((error) => {
        console.error("Erreur chargement Prophet :", error);
      });
  }, []);

  return (
    <div style={{ marginTop: '40px' }}>
      <h2>🔮 Prédiction Prophet</h2>
      {chartData ? (
        <Line
          data={chartData}
          options={{
            responsive: true,
            plugins: {
              legend: { position: 'top' },
              title: { display: true, text: 'Prévision Prophet vs Réel' },
            },
            scales: {
              x: { display: true, title: { display: true, text: 'Date/Heure' } },
              y: { display: true, title: { display: true, text: 'Consommation (MW)' } },
            },
          }}
        />
      ) : (
        <p>Chargement...</p>
        
      )}

    {chartData && (
    <div className="model-scores">
        <h4>📊 Scores Prophet</h4>
        <ul>
        <li>RMSE : {scores.rmse}</li>
        <li>MAE : {scores.mae}</li>
        <li>MAPE : {scores.mape} %</li>
        </ul>
    </div>
    )}

    </div>
  );
}

export default ProphetChart;
