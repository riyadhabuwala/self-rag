import React, { useState, useEffect, useContext } from 'react';
import { API } from '../api';
import { AppContext } from '../App';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function Metrics() {
  const { addToast } = useContext(AppContext);
  const [metrics, setMetrics] = useState({});
  const [cacheStats, setCacheStats] = useState({});

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      const [m, c] = await Promise.all([
        API.metrics(),
        API.cacheStats().catch(() => ({}))
      ]);
      setMetrics(m);
      setCacheStats(c);
    } catch (e) {
      addToast(e.message, 'error');
    }
  };

  const chartData = {
    labels: ['High', 'Medium', 'Low'],
    datasets: [{
      data: [
        metrics.confidence_distribution?.high || 0,
        metrics.confidence_distribution?.medium || 0,
        metrics.confidence_distribution?.low || 0
      ],
      backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
      borderWidth: 0
    }]
  };

  const totalQueries = metrics.total_queries || 0;
  const cacheHitRate = totalQueries > 0 ? ((metrics.cache_hits / totalQueries) * 100).toFixed(1) : 0;
  const hallucinationRate = totalQueries > 0 ? ((metrics.hallucinations_caught / totalQueries) * 100).toFixed(1) : 0;

  return (
    <>
      <div className="docs-header">
        <h2>System Metrics</h2>
        <button className="btn-outline" onClick={loadMetrics}>Refresh</button>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="text-sm text-secondary font-medium tracking-wider">Total Queries</div>
          <div className="metric-value">{totalQueries}</div>
        </div>
        <div className="metric-card">
          <div className="text-sm text-secondary font-medium tracking-wider">Cache Hit Rate</div>
          <div className="metric-value">{cacheHitRate}%</div>
        </div>
        <div className="metric-card">
          <div className="text-sm text-secondary font-medium tracking-wider">Hallucinations Caught</div>
          <div className="metric-value">{hallucinationRate}%</div>
        </div>
      </div>

      <div className="charts-row">
        <div className="chart-card">
          <h4 className="text-sm text-secondary font-medium tracking-wider">Confidence Distribution</h4>
          <div className="chart-container">
            <Doughnut data={chartData} options={{ maintainAspectRatio: false, cutout: '70%', plugins: { legend: { position: 'right', labels: { color: '#94a3b8' } } } }} />
          </div>
        </div>
        <div className="chart-card" style={{ justifyContent: 'space-between' }}>
          <div>
            <h4 className="text-sm text-secondary font-medium tracking-wider" style={{ marginBottom: '24px' }}>Performance Averages</h4>
            <div style={{ marginBottom: '24px' }}>
              <div className="text-xs text-muted mb-1 tracking-wider">Avg Usefulness Score</div>
              <div className="text-xl font-bold flex items-center gap-4 text-white">
                <span>{(metrics.avg_usefulness_score || 0).toFixed(1)} / 5</span>
              </div>
            </div>
            <div>
              <div className="text-xs text-muted mb-1 tracking-wider">Avg Response Time</div>
              <div className="flex flex-col gap-3" style={{ marginTop: '12px' }}>
                <div className="flex items-center gap-3 text-sm text-white">
                  <span>⚡</span> <span className="font-mono font-bold">{Math.round(metrics.avg_response_time_ms_pipeline || 0)}ms</span> <span className="text-muted">(pipeline)</span>
                </div>
                <div className="flex items-center gap-3 text-sm text-success">
                  <span>💾</span> <span className="font-mono font-bold">{Math.round(metrics.avg_response_time_ms_cache || 0)}ms</span> <span className="text-muted">(cache hit)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
