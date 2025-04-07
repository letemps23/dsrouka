function DownloadLinks() {
    const files = [
      "consommation-filtree",
      "dataset-corrige",
      "dataset-prediction",
      "resultats-prediction",
      "courbe",
      "decomposition",
      "pics-creux",
      "courbe-evolution-30min"
    ];
  
    return (
      <div className="download-section">
        
        <h3>📂 Fichiers à télécharger :</h3>
        <ul>
          {files.map(name => (
            <li key={name}>
              <a href={`${process.env.REACT_APP_API_URL}/telecharger/${name}`} target="_blank" rel="noopener noreferrer">
                {name.replace(/-/g, ' ').toUpperCase()}
              </a>
            </li>
          ))}
        </ul>
      </div>
    );
  }
  
  export default DownloadLinks;
  