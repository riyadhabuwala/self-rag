import React, { useState, useEffect, useRef, useContext } from 'react';
import { API } from '../api';
import { AppContext } from '../App';
import { marked } from 'marked';

export default function Chat() {
  const { currentSessionId, setCurrentSessionId, addToast } = useContext(AppContext);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isPending, setIsPending] = useState(false);
  const [sessions, setSessions] = useState([]);
  const chatRef = useRef(null);

  const [filters, setFilters] = useState({ ticker: '', fiscal_year: '', doc_type: '' });

  useEffect(() => {
    loadSessionsDropdown();
  }, []);

  useEffect(() => {
    if (currentSessionId) {
      loadSessionMessages(currentSessionId);
    } else {
      setMessages([{ role: 'assistant', content: 'Hello! I am your **Financial Intelligence Assistant**.\n\nI can analyze your indexed documents using a self-reflective RAG pipeline. Ask me anything about revenues, risks, margins, or operations.' }]);
    }
  }, [currentSessionId]);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, isPending]);

  const loadSessionsDropdown = async () => {
    try {
      const data = await API.sessions(false);
      if (Array.isArray(data)) setSessions(data);
    } catch (e) {}
  };

  const loadSessionMessages = async (id) => {
    try {
      const data = await API.getSession(id);
      const sessionData = typeof data === 'string' ? JSON.parse(data) : data;
      if (sessionData.messages) {
        setMessages(sessionData.messages);
      }
    } catch (e) {
      addToast('Failed to load session', 'error');
    }
  };

  const startNewSession = async () => {
    try {
      const res = await API.createSession({ title: "New Session" });
      setCurrentSessionId(res.session_id);
      setMessages([{ role: 'assistant', content: 'Started a new conversation session.' }]);
      loadSessionsDropdown();
    } catch (e) {
      addToast(e.message, 'error');
    }
  };

  const submitQuery = async () => {
    if (!input.trim() || isPending) return;
    const q = input.trim();
    setInput('');
    setIsPending(true);

    const userMsg = { role: 'user', content: q };
    setMessages(prev => [...prev, userMsg]);

    const activeFilters = {};
    if (filters.ticker) activeFilters.ticker = filters.ticker.toUpperCase();
    if (filters.fiscal_year) activeFilters.fiscal_year = filters.fiscal_year;
    if (filters.doc_type) activeFilters.doc_type = filters.doc_type;

    try {
      const req = { query: q };
      if (Object.keys(activeFilters).length > 0) req.filters = activeFilters;
      if (currentSessionId) req.session_id = currentSessionId;

      const res = await API.query(req);
      if (res.session_id && res.session_id !== currentSessionId) {
        setCurrentSessionId(res.session_id);
        loadSessionsDropdown();
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.answer,
        meta: {
          confidence: res.confidence,
          groundedness: res.groundedness,
          usefulness_score: res.usefulness_score,
          unsupported_claims: res.unsupported_claims,
          sources: res.sources,
          response_time_ms: res.response_time_ms,
          retries: res.retries,
          cache_hit: res.cache_hit
        }
      }]);
    } catch (e) {
      addToast(e.message, 'error');
      setMessages(prev => [...prev, { role: 'assistant', content: `**Error:** ${e.message}` }]);
    } finally {
      setIsPending(false);
    }
  };

  const renderMessageContent = (role, content) => {
    if (role === 'user') return <div dangerouslySetInnerHTML={{ __html: content }} />;
    return <div dangerouslySetInnerHTML={{ __html: marked.parse(content) }} />;
  };

  return (
    <>
      <div className="chat-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <h3 className="text-lg">AI Assistant</h3>
          <select className="select-input" style={{ width: 'auto', minWidth: '200px' }} value={currentSessionId} onChange={e => setCurrentSessionId(e.target.value)}>
            <option value="">Current Session</option>
            {sessions.map(s => <option key={s.session_id} value={s.session_id}>{s.title || s.session_id}</option>)}
          </select>
        </div>
        <button className="btn-outline" onClick={startNewSession}>✨ New Session</button>
      </div>

      <div id="chat-messages" ref={chatRef}>
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-content markdown-body">
              {renderMessageContent(msg.role, msg.content)}
            </div>
            {msg.meta && (
              <>
                <div className="badges-row">
                  {msg.meta.confidence && <span className={`badge ${msg.meta.confidence === 'high' ? 'badge-green' : 'badge-amber'}`}>{msg.meta.confidence.toUpperCase()} CONFIDENCE</span>}
                  {msg.meta.groundedness && <span className={`badge ${msg.meta.groundedness === 'fully' ? 'badge-blue' : 'badge-red'}`}>{msg.meta.groundedness.toUpperCase()} GROUNDED</span>}
                  {msg.meta.usefulness_score !== undefined && (
                    <div className="star-rating" style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px' }}>
                      <span className="text-muted font-medium ml-2">{msg.meta.usefulness_score}/5 Usefulness</span>
                      {'★'.repeat(msg.meta.usefulness_score).padEnd(5, '☆').split('').map((s, idx) => s === '★' ? <span key={idx}>★</span> : <span key={idx} className="empty">★</span>)}
                    </div>
                  )}
                </div>
                {msg.meta.unsupported_claims?.length > 0 && (
                  <div className="hallucination-warning">
                    <div className="hallucination-warning-title">⚠️ {msg.meta.unsupported_claims.length} claim(s) could not be verified:</div>
                    <ul>{msg.meta.unsupported_claims.map((c, idx) => <li key={idx}>{c}</li>)}</ul>
                  </div>
                )}
                {msg.meta.sources?.length > 0 && (
                  <div className="sources-container">
                    <div className="sources-header">📄 Sources ({msg.meta.sources.length})</div>
                  </div>
                )}
                <div className="message-meta-footer">
                  <span>⏱ {msg.meta.response_time_ms || 0}ms</span>
                  <span>🔄 {msg.meta.retry_count || msg.meta.retries || 0} retries</span>
                  <span>💾 cached: {msg.meta.cache_hit ? <span className="text-success">Yes</span> : 'No'}</span>
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      {isPending && (
        <div className="typing-indicator">
          <div className="dot"></div><div className="dot"></div><div className="dot"></div>
          <span style={{ marginLeft: '12px', fontWeight: 500 }}>Running Self-RAG pipeline...</span>
        </div>
      )}

      <div className="chat-input-container">
        <div className="chat-filters">
          <div className="filter-group">
            <label>Ticker:</label>
            <input type="text" className="input-text" placeholder="e.g. AAPL" style={{ width: '90px', padding: '4px 12px' }} value={filters.ticker} onChange={e => setFilters({ ...filters, ticker: e.target.value })} />
          </div>
          <div className="filter-group">
            <label>Year:</label>
            <select className="select-input" style={{ width: '110px', padding: '4px 12px' }} value={filters.fiscal_year} onChange={e => setFilters({ ...filters, fiscal_year: e.target.value })}>
              <option value="">Any</option><option value="FY2024">FY2024</option><option value="FY2023">FY2023</option><option value="FY2022">FY2022</option>
            </select>
          </div>
          <div className="filter-group">
            <label>Type:</label>
            <select className="select-input" style={{ width: '150px', padding: '4px 12px' }} value={filters.doc_type} onChange={e => setFilters({ ...filters, doc_type: e.target.value })}>
              <option value="">Any</option><option value="10-K">10-K</option><option value="10-Q">10-Q</option><option value="earnings-transcript">Earnings Transcript</option>
            </select>
          </div>
        </div>

        <div className="input-box">
          <textarea placeholder="Ask a question about your financial documents... (Shift+Enter for newline)" value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitQuery(); } }} disabled={isPending}></textarea>
          <button className="btn-primary" title="Send Query" onClick={submitQuery} disabled={isPending}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
          </button>
        </div>
        <div className="input-footer">
          <span>Markdown supported</span>
          <span style={{ color: input.length > 1000 ? 'var(--danger)' : 'var(--text-muted)' }}>{1000 - input.length}</span>
        </div>
      </div>
    </>
  );
}
