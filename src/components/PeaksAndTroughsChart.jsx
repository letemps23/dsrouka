// src/components/PeaksAndTroughsChart.jsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';

function PeaksAndTroughsChart() {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get(`${process.env.REACT_APP_API_URL}/api/pics-creux-json`)
      .then(res => setData(res.data))
      .catch(err => console.error("Erreur chargement pics/creux :", err));
  }, []);

  if (!data) return <p>Chargement des pics et creux...</p>;

  const options = {
    responsive: true,
    plugins: { legend: { position: 'top' } }
  };

  const chartData = {
    labels: data.date,
    datasets: [
      {
        label: "Pic journalier",
        data: data.pics,
        borderColor: 'orange',
        fill: false,
        tension: 0.3,
      },
      {
        label: "Creux journalier",
        data: data.creux,
        borderColor: 'purple',
        fill: false,
        tension: 0.3,
      },
    ],
  };

  return (
    <div className="chart-section">
      <h3>📊 Pics & Creux Journaliers</h3>
      <Line data={chartData} options={options} />
    </div>
  );
}

export default PeaksAndTroughsChart;
