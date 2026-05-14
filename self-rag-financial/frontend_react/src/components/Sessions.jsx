import React, { useState, useEffect, useContext } from 'react';
import { API } from '../api';
import { AppContext } from '../App';

export default function Sessions() {
  const { setActiveTab, setCurrentSessionId, addToast } = useContext(AppContext);
  const [sessions, setSessions] = useState([]);
  const [showArchived, setShowArchived] = useState(false);

  useEffect(() => {
    loadSessions();
  }, [showArchived]);

  const loadSessions = async () => {
    try {
      const data = await API.sessions(showArchived);
      setSessions(Array.isArray(data) ? data : []);
    } catch (e) {
      addToast(e.message, 'error');
    }
  };

  const archiveSession = async (id) => {
    try {
      await API.archiveSession(id);
      addToast('Session archived', 'success');
      loadSessions();
    } catch (e) {
      addToast(e.message, 'error');
    }
  };

  return (
    <>
      <div className="docs-header">
        <div>
          <h2>Conversation History</h2>
          <p className="text-sm text-secondary" style={{ marginTop: '8px' }}>Manage your past analysis sessions.</p>
        </div>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-secondary font-medium" style={{ cursor: 'pointer' }}>
            <input type="checkbox" checked={showArchived} onChange={e => setShowArchived(e.target.checked)} /> Show Archived
          </label>
        </div>
      </div>
      <div className="sessions-list">
        {sessions.map(s => (
          <div key={s.session_id} className="session-card" style={{ opacity: s.archived ? 0.6 : 1 }}>
            <div className="session-info">
              <h3>{s.title || 'Untitled Session'} {s.archived ? '(Archived)' : ''}</h3>
              <div className="session-meta">
                Created: {s.created_at ? new Date(s.created_at).toLocaleDateString() : 'Unknown'} • {s.messages?.length || s.message_count || 0} messages
              </div>
            </div>
            <div className="session-actions">
              <button className="btn-outline" onClick={() => { setCurrentSessionId(s.session_id); setActiveTab('chat'); }}>Open</button>
              {!s.archived && (
                <button className="btn-outline" onClick={() => archiveSession(s.session_id)} style={{ color: 'var(--danger)', borderColor: 'rgba(239,68,68,0.3)' }}>Archive</button>
              )}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
