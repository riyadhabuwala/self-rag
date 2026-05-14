import React, { useState, useEffect } from 'react';
import { API } from './api';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import Documents from './components/Documents';
import Sessions from './components/Sessions';
import Metrics from './components/Metrics';
import ApiModal from './components/ApiModal';
import ToastContainer from './components/ToastContainer';

export const AppContext = React.createContext();

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [isConnected, setIsConnected] = useState(false);
  const [showApiModal, setShowApiModal] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState('');
  const [statusText, setStatusText] = useState('System connected');
  const [statusClass, setStatusClass] = useState('ok');
  
  const [theme, setTheme] = useState(localStorage.getItem('selfrag_theme') || 'dark');

  const addToast = (message, type = 'success') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  };

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('selfrag_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  useEffect(() => {
    if (!API.apiKey()) {
      setShowApiModal(true);
    } else {
      checkHealth();
      const interval = setInterval(checkHealth, 30000);
      return () => clearInterval(interval);
    }
  }, []);

  const checkHealth = async () => {
    try {
      await API.health();
      setIsConnected(true);
      setStatusClass('ok');
    } catch (e) {
      setStatusClass('error');
    }
  };

  const handleConnect = async (url, key) => {
    localStorage.setItem('selfrag_api_url', url);
    localStorage.setItem('selfrag_api_key', key);
    try {
      const res = await API.health();
      if (res.status === 'ok') {
        addToast(`Connected — ${res.chroma_doc_count} documents indexed`, 'success');
        setShowApiModal(false);
        setIsConnected(true);
      }
    } catch (err) {
      addToast(`Connection failed: ${err.message}`, 'error');
    }
  };

  return (
    <AppContext.Provider value={{
      activeTab, setActiveTab,
      currentSessionId, setCurrentSessionId,
      addToast,
      isConnected,
      theme, toggleTheme
    }}>
      <div id="app-container" className={!isConnected && showApiModal ? 'hidden' : ''}>
        <Sidebar statusClass={statusClass} onSettingsClick={() => setShowApiModal(true)} />
        
        <main id="main-content">
          <div className={`tab-pane ${activeTab === 'chat' ? 'active' : ''}`} id="tab-chat">
            <Chat />
          </div>
          <div className={`tab-pane ${activeTab === 'documents' ? 'active' : ''}`} id="tab-documents">
            <Documents />
          </div>
          <div className={`tab-pane ${activeTab === 'sessions' ? 'active' : ''}`} id="tab-sessions">
            <Sessions />
          </div>
          <div className={`tab-pane ${activeTab === 'metrics' ? 'active' : ''}`} id="tab-metrics">
            <Metrics />
          </div>
        </main>
      </div>

      {showApiModal && <ApiModal onConnect={handleConnect} />}
      <ToastContainer toasts={toasts} />
    </AppContext.Provider>
  );
}

export default App;
