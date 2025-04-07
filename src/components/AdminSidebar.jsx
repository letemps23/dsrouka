// src/components/AdminSidebar.jsx
import React from 'react';
import './AdminSidebar.css';

function AdminSidebar() {
  const links = [
    { name: "Fichiers à télécharger", sublinks: [
      { label: "Consommation Filtrée", path: "/telecharger/consommation-filtree" },
      { label: "Dataset Corrigé", path: "/telecharger/dataset-corrige" },
      { label: "Dataset Prédiction", path: "/telecharger/dataset-prediction" },
      { label: "Résultats LSTM", path: "/telecharger/resultats-prediction" },
      { label: "Courbe LSTM", path: "/telecharger/courbe" },
      { label: "Décomposition", path: "/telecharger/decomposition" },
      { label: "Pics & Creux", path: "/telecharger/pics-creux" },
      { label: "Courbe Interpolée", path: "/telecharger/courbe-evolution-30min" },
    ]}
  ];

  return (
    <aside className="admin-sidebar">
      <h2>⚙️ Dashboard LSTM</h2>
      <nav>
        {links.map((section, idx) => (
          <div key={idx} className="sidebar-section">
            <h4>{section.name}</h4>
            <ul>
              {section.sublinks.map((sublink, i) => (
                <li key={i}>
                  <a href={`http://localhost:8000${sublink.path}`} target="_blank" rel="noreferrer">
                    {sublink.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
    </aside>
  );
}

export default AdminSidebar;
