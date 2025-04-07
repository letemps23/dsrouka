// src/pages/DataPreparationPage.jsx
import React, { useState } from 'react';
import UploadForm from '../components/UploadForm';
import ActionButtons from '../components/ActionButtons';
import './DataPreparationPage.css';

function DataPreparationPage() {
  const [refreshFlag, setRefreshFlag] = useState(false);
  const triggerRefresh = () => setRefreshFlag(prev => !prev);

  return (
    <div className="data-page">
      <h2>📤 Upload et Préparation des Données</h2>

      <div className="upload-block">
        <UploadForm />
      </div>

      <div className="buttons-block">
        <ActionButtons onRefresh={triggerRefresh} />
      </div>
    </div>
  );
}

export default DataPreparationPage;
