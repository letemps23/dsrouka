import React, { useState } from 'react';
import axios from 'axios';

function UploadForm() {
  const [file, setFile] = useState(null);
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    await axios.post(`${process.env.REACT_APP_API_URL}/upload/`, formData);
    alert("Fichier envoyé avec succès !");
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="file" onChange={e => setFile(e.target.files[0])} />
      <button type="submit">Charger le fichier</button>
    </form>
  );
}

export default UploadForm;
