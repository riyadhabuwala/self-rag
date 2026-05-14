import React from 'react';

export default function ToastContainer({ toasts }) {
  return (
    <div id="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast ${t.type}`}>
          <span>{t.type === 'error' ? '❌' : (t.type === 'warning' ? '⚠️' : '✅')} &nbsp; {t.message}</span>
        </div>
      ))}
    </div>
  );
}
