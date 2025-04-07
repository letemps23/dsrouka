import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';

function ArimaChart() {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    axios.get(`${process.env.REACT_APP_API_URL}/api/arima-json`)
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
              label: 'Prévisions ARIMA',
              data: data.prevision,
              borderColor: 'orange',
              borderDash: [5, 5],
              fill: false,
            },
          ],
        });
      })
      .catch((error) => {
        console.error("Erreur de chargement ARIMA :", error);
      });
  }, []);

  return (
    <div>
      <h2>📈 Prédiction ARIMA</h2>
      {chartData ? (
        <Line
          data={chartData}
          options={{
            responsive: true,
            plugins: {
              legend: { position: 'top' },
              title: { display: true, text: 'Prévision ARIMA vs Réel' },
            },
            scales: {
              x: { display: true, title: { display: true, text: 'Date' } },
              y: { display: true, title: { display: true, text: 'Consommation (MW)' } },
            },
          }}
        />
      ) : (
        <p>Chargement...</p>
      )}
    </div>
  );
}

export default ArimaChart;
