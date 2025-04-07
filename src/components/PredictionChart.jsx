import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, TimeScale
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import ArimaChart from '../components/ArimaChart';
import ProphetChart from '../components/ProphetChart';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, TimeScale);

function PredictionChart({ type = "prediction" }) {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const url = type === "prediction"
        ? `${process.env.REACT_APP_API_URL}/api/prediction-json`
        : `${process.env.REACT_APP_API_URL}/api/interpolation-json`;

      try {
        const res = await axios.get(url);
        const data = res.data;

        const labels = data.map(row => row.DATETIME);
        const dataset1 = type === "prediction"
          ? data.map(row => row["Valeur Réelle"])
          : data.map(row => row["CONSOMMATION_TOTALE"]);
        const dataset2 = type === "prediction" ? data.map(row => row["Prédiction LSTM"]) : null;

        const datasets = [
          {
            label: type === "prediction" ? "Valeur Réelle" : "Consommation Interpolée",
            data: dataset1,
            borderColor: type === "prediction" ? "blue" : "green",
            fill: false,
            tension: 0.2,
          }
        ];

        if (dataset2) {
          datasets.push({
            label: "Prédiction LSTM",
            data: dataset2,
            borderColor: "red",
            fill: false,
            tension: 0.2,
          });
        }

        setChartData({
          labels,
          datasets
        });
      } catch (error) {
        console.error("Erreur de chargement des données :", error);
      }
    };

    fetchData();
  }, [type]);

  return (
    <div style={{ marginTop: "2rem" }}>
      <h2>{type === "prediction" ? "📈 Prédiction LSTM" : "📊 Consommation Interpolée"}</h2>
      {chartData ? (
        <Line
          data={chartData}
          options={{
            responsive: true,
            scales: {
              x: {
                type: 'time',
                time: { unit: 'day' },
                title: { display: true, text: 'Date' }
              },
              y: {
                title: { display: true, text: 'MW' }
              }
            }
          }}
        />
      ) : (
        <p>Chargement des données...</p>
      )}
      <ArimaChart />
      <ProphetChart />
    </div>
  );
}

export default PredictionChart;
