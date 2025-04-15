import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, LineElement, TimeScale, LinearScale,
  Title, Tooltip, Legend, PointElement
} from 'chart.js';
import 'chartjs-adapter-date-fns';

ChartJS.register(LineElement, TimeScale, LinearScale, Title, Tooltip, Legend, PointElement);

function ConsumptionChart() {
  const [chartData, setChartData] = useState(null);
  const [selectedYear, setSelectedYear] = useState(2019);
  const [selectedMonth, setSelectedMonth] = useState("1"); // Janvier par défaut

  const fetchData = useCallback(async () => {
    const url = `${process.env.REACT_APP_API_URL}/api/consommation-json?annee=${selectedYear}&mois=${selectedMonth}`;
    try {
      const res = await axios.get(url);
      const rawData = res.data;

      // ✅ Nettoyage des données
      const cleanData = rawData.filter(d =>
        d.CONSOMMATION_TOTALE > 0 &&
        !isNaN(d.CONSOMMATION_TOTALE) &&
        d.DATETIME
      );

      console.log("Données filtrées :", cleanData.length);

      setChartData({
        labels: cleanData.map(d => d.DATETIME),
        datasets: [{
          label: 'Consommation (MW)',
          data: cleanData.map(d => d.CONSOMMATION_TOTALE),
          borderColor: 'blue',
          backgroundColor: 'rgba(0, 0, 255, 0.2)',
          tension: 0.3,
          pointRadius: 2,
          spanGaps: false,
          animation: false,
          showLine: false
        }]
      });
    } catch (err) {
      console.error("Erreur lors du chargement :", err);
    }
  }, [selectedYear, selectedMonth]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return (
    <div>
      <h3>📆 Filtrer par année et mois</h3>
      <div style={{ display: 'flex', gap: '1rem' }}>
        <select value={selectedYear} onChange={e => setSelectedYear(parseInt(e.target.value))}>
          {[2014, 2015, 2016, 2017, 2018, 2019].map((year) => (
            <option key={year} value={year}>{year}</option>
          ))}
        </select>

        <select value={selectedMonth} onChange={e => setSelectedMonth(e.target.value)}>
          <option value="1">Janvier</option>
          <option value="2">Février</option>
          <option value="3">Mars</option>
          <option value="4">Avril</option>
          <option value="5">Mai</option>
          <option value="6">Juin</option>
          <option value="7">Juillet</option>
          <option value="8">Août</option>
          <option value="9">Septembre</option>
          <option value="10">Octobre</option>
          <option value="11">Novembre</option>
          <option value="12">Décembre</option>
        </select>
      </div>

      <div style={{ marginTop: '20px' }}>
        {chartData ? (
          <Line
            data={chartData}
            options={{
              responsive: true,
              plugins: {
                legend: { position: 'top' },
                title: {
                  display: true,
                  text: `Consommation pour ${selectedYear} - ${selectedMonth.padStart(2, '0')}`
                }
              },
              spanGaps: false,
              scales: {
                x: {
                  type: 'time',
                  time: { unit: 'day' },
                  title: { display: true, text: 'Date' }
                },
                y: {
                  title: { display: true, text: 'Consommation (MW)' }
                }
              }
            }}
          />
        ) : (
          <p>Chargement...</p>
        )}
      </div>
    </div>
  );
}

export default ConsumptionChart;
