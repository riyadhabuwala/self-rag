import React, { useState, useEffect, useContext } from 'react';
import { API } from '../api';
import { AppContext } from '../App';

export default function Documents() {
  const { addToast } = useContext(AppContext);
  const [docs, setDocs] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadData, setUploadData] = useState({ file: null, ticker: '', type: '10-K', year: '' });

  useEffect(() => {
    loadDocs();
  }, []);

  const loadDocs = async () => {
    try {
      const data = await API.docsInfo();
      setDocs(data.documents || Object.values(data));
    } catch (e) {
      addToast(e.message, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!uploadData.file) return addToast('Select a file', 'warning');
    const formData = new FormData();
    formData.append('file', uploadData.file);
    const params = new URLSearchParams();
    if (uploadData.ticker) params.append('ticker', uploadData.ticker.toUpperCase());
    if (uploadData.type) params.append('doc_type', uploadData.type);
    if (uploadData.year) params.append('fiscal_year', uploadData.year);
    params.append('force', 'true');

    try {
      await API.upload(`/ingest?${params.toString()}`, formData);
      addToast('Ingestion started!', 'success');
      setShowUpload(false);
      setTimeout(loadDocs, 5000);
    } catch (err) {
      addToast(err.message, 'error');
    }
  };

  const totalChunks = docs.reduce((acc, d) => acc + (d.chunk_count || 0), 0);

  return (
    <>
      <div className="docs-header">
        <div>
          <h2>Documents Library</h2>
          <p className="text-sm text-secondary" style={{ marginTop: '8px' }}>
            {isLoading ? 'Loading...' : `${docs.length} documents | ${totalChunks} chunks indexed`}
          </p>
        </div>
        <button className="btn-primary" onClick={() => setShowUpload(true)}>Upload Document</button>
      </div>

      <div className="docs-grid">
        {docs.map((d, i) => (
          <div key={i} className="doc-card">
            <div className="doc-card-header">
              <div className="flex gap-2">
                <span className="badge badge-gray">{d.ticker}</span>
                <span className="badge badge-gray">{d.doc_type}</span>
                <span className="badge badge-gray">{d.fiscal_year}</span>
              </div>
              <span className="chunk-badge">{d.chunk_count || 0} chunks</span>
            </div>
            <div className="doc-title">{d.source_file || `${d.ticker} ${d.doc_type}`}</div>
            <div className="doc-card-footer">
              <span>Ingested: {d.ingested_at ? new Date(d.ingested_at).toLocaleDateString() : 'Unknown'}</span>
            </div>
          </div>
        ))}
      </div>

      {showUpload && (
        <div className="slide-panel-overlay open" onClick={(e) => { if (e.target === e.currentTarget) setShowUpload(false); }}>
          <div className="slide-panel open">
            <div className="slide-panel-header">
              <h3>Upload Document</h3>
              <button className="btn-ghost" onClick={() => setShowUpload(false)}>✕</button>
            </div>
            <div className="slide-panel-body">
              <form onSubmit={handleUpload}>
                <div className="drop-zone" style={{ marginBottom: '24px' }}>
                  <div style={{ fontSize: '40px', marginBottom: '12px' }}>📄</div>
                  <div style={{ fontWeight: 600, color: '#fff' }}>Select PDF or HTML</div>
                  <input type="file" onChange={e => setUploadData({ ...uploadData, file: e.target.files[0] })} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  <div>
                    <label className="text-xs text-secondary font-medium mb-1 block uppercase">Ticker</label>
                    <input type="text" className="input-text" required value={uploadData.ticker} onChange={e => setUploadData({ ...uploadData, ticker: e.target.value })} />
                  </div>
                  <div>
                    <label className="text-xs text-secondary font-medium mb-1 block uppercase">Doc Type</label>
                    <select className="select-input" value={uploadData.type} onChange={e => setUploadData({ ...uploadData, type: e.target.value })}>
                      <option value="10-K">10-K</option><option value="10-Q">10-Q</option><option value="earnings-transcript">Earnings Transcript</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-secondary font-medium mb-1 block uppercase">Fiscal Year</label>
                    <input type="text" className="input-text" required value={uploadData.year} onChange={e => setUploadData({ ...uploadData, year: e.target.value })} />
                  </div>
                </div>
                <button type="submit" className="btn-primary" style={{ width: '100%', marginTop: '32px' }}>Upload & Ingest</button>
              </form>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
