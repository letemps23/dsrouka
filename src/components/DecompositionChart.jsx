// src/components/DecompositionChart.jsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';

function DecompositionChart() {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    axios.get(`${process.env.REACT_APP_API_URL}/api/decomposition-json`)
      .then(res => setChartData(res.data))
      .catch(err => console.error("Erreur chargement décomposition :", err));
  }, []);

  if (!chartData) return <p>Chargement des courbes de décomposition...</p>;

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false }
    }
  };

  const makeChart = (label, data, color) => ({
    labels: chartData.datetime,
    datasets: [{
      label,
      data,
      borderColor: color,
      borderWidth: 2,
      fill: false
    }]
  });

  return (
    <div className="chart-section">
      <h3>📈 Tendance</h3>
      <Line data={makeChart("Tendance", chartData.trend, 'blue')} options={options} />

      <h3>🔁 Saisonnalité</h3>
      <Line data={makeChart("Saisonnalité", chartData.seasonal, 'green')} options={options} />

      <h3>⚠️ Résidus</h3>
      <Line data={makeChart("Résidus", chartData.resid, 'red')} options={options} />
    </div>
  );
}

export default DecompositionChart;
