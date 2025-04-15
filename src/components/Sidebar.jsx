import React, { useRef, useState } from 'react';
import './Sidebar.css';

function Sidebar() {
  const sidebarRef = useRef(null);
  const [isResizing, setIsResizing] = useState(false);

  const startResizing = (e) => {
    setIsResizing(true);
    document.addEventListener('mousemove', resize);
    document.addEventListener('mouseup', stopResizing);
  };

  const resize = (e) => {
    if (isResizing && sidebarRef.current) {
      const newWidth = Math.max(180, Math.min(e.clientX, 400));
      sidebarRef.current.style.width = `${newWidth}px`;
    }
  };

  const stopResizing = () => {
    setIsResizing(false);
    document.removeEventListener('mousemove', resize);
    document.removeEventListener('mouseup', stopResizing);
  };

  const links = [
    { name: "Accueil", path: "/" },
    { name: "Préparation des données", path: "/preparation" },
    { name: "Prédiction", path: "/prediction" },
    //{ name: "Consommation", path: "/interpolated" },
    { name: "Consommation (graphique)", path: "/consommation-graphique" },
    //{ name: "Décomposition", path: "/decomposition" },
    //{ name: "Pics & Creux", path: "/pics-creux" },
    { name: "Téléchargement", path: "/telechargement" }
  ];

  return (
    <div className="resizable-sidebar" ref={sidebarRef}>
      <h2 className="sidebar-title">⚡ Prédiction  Energy</h2>
      <nav>
        <ul>
          {links.map((link, i) => (
            <li key={i}>
              <a href={link.path}>{link.name}</a>
            </li>
          ))}
        </ul>
      </nav>
      <div className="resizer" onMouseDown={startResizing} />
    </div>
  );
}

export default Sidebar;
