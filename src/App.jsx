import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import DataPreparationPage from './pages/DataPreparationPage';
import PredictionChart from './components/PredictionChart';
import DownloadLinks from './components/DownloadLinks';
import DecompositionChart from './components/DecompositionChart';
import PeaksAndTroughsChart from './components/PeaksAndTroughsChart';
import ArimaChart from './components/ArimaChart';
import './App.css';

function App() {
  return (
    <Router>
      <Header />
      <div className="main-wrapper">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<h2>Bienvenue sur le Dashboard</h2>} />
            <Route path="/preparation" element={<DataPreparationPage />} />
            <Route path="/prediction" element={<PredictionChart type="prediction" />} />
            <Route path="/interpolated" element={<PredictionChart type="interpolated" />} />
            <Route path="/telechargement" element={<DownloadLinks />} />
            <Route path="/arima" element={<ArimaChart />} />
            {/* ✅ Nouvelles routes */}
            <Route path="/decomposition" element={<DecompositionChart />} />
            <Route path="/pics-creux" element={<PeaksAndTroughsChart />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
