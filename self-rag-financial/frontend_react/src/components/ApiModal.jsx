import React, { useState } from 'react';

export default function ApiModal({ onConnect }) {
  const [url, setUrl] = useState(localStorage.getItem('selfrag_api_url') || 'http://localhost:8000');
  const [key, setKey] = useState(localStorage.getItem('selfrag_api_key') || '');

  return (
    <div className="modal-overlay open">
      <div className="modal">
        <h3>🔐 Connect to Self-RAG API</h3>
        <p className="text-sm text-secondary">Provide your backend connection details.</p>
        <div>
          <label className="text-xs text-muted font-medium mb-1 block">API Base URL</label>
          <input type="text" className="input-text" value={url} onChange={e => setUrl(e.target.value)} />
        </div>
        <div>
          <label className="text-xs text-muted font-medium mb-1 block">API Key</label>
          <input type="password" className="input-text" value={key} onChange={e => setKey(e.target.value)} placeholder="dev-key-change-in-production" />
        </div>
        <button className="btn-primary" onClick={() => onConnect(url, key)}>Connect</button>
      </div>
    </div>
  );
}
