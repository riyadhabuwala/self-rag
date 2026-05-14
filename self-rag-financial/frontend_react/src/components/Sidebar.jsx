import React, { useContext, useEffect, useState } from 'react';
import { AppContext } from '../App';
import { API } from '../api';

export default function Sidebar({ statusClass, onSettingsClick }) {
  const { activeTab, setActiveTab, currentSessionId, setCurrentSessionId, theme, toggleTheme } = useContext(AppContext);
  const [sessions, setSessions] = useState([]);

  useEffect(() => {
    loadSessions();
  }, [currentSessionId, activeTab]);

  const loadSessions = async () => {
    try {
      const data = await API.sessions(false);
      if (Array.isArray(data)) setSessions(data.slice(0, 10));
    } catch (e) { console.error(e); }
  };

  const navItems = [
    { id: 'chat', icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>, label: 'Chat' },
    { id: 'documents', icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>, label: 'Documents' },
    { id: 'sessions', icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>, label: 'Sessions' },
    { id: 'metrics', icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>, label: 'Metrics' }
  ];

  return (
    <aside id="sidebar">
      <div className="sidebar-header">
        <span>✦ Intelligence</span>
        <div style={{display:'flex', alignItems:'center', gap:'12px'}}>
          <div className={`status-dot ${statusClass}`}></div>
          <button className="btn-ghost" style={{padding:'4px'}} onClick={toggleTheme} title="Toggle Theme">
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          <button className="btn-ghost" style={{padding:'4px'}} onClick={onSettingsClick} title="Settings">⚙</button>
        </div>
      </div>
      <nav className="nav-menu">
        {navItems.map(item => (
          <a key={item.id} href="#" className={`nav-item ${activeTab === item.id ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveTab(item.id); }}>
            {item.icon} {item.label}
          </a>
        ))}
      </nav>
      <div className="recent-sessions">
        <div className="recent-sessions-title">Recent Sessions</div>
        <div style={{display:'flex', flexDirection:'column', gap:'4px'}}>
          {sessions.map(s => (
            <div key={s.session_id} className="session-item" onClick={() => { setCurrentSessionId(s.session_id); setActiveTab('chat'); }}>
              {s.title || `Session ${s.session_id.substring(0,8)}`}
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
