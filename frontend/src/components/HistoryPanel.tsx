import React from 'react';
import { Clock, FileText, ChevronRight } from 'lucide-react';
import './HistoryPanel.css';

interface HistoryItem {
  id: number;
  text: string;
  timestamp: string;
  results: any;
}

interface HistoryPanelProps {
  history: HistoryItem[];
  onSelectItem: (item: HistoryItem) => void;
}

export const HistoryPanel: React.FC<HistoryPanelProps> = ({ history, onSelectItem }) => {
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return date.toLocaleDateString();
  };

  const getConfidenceColor = (results: any) => {
    if (!results?.overall_confidence) return '#64748b';
    const confidence = results.overall_confidence;
    if (confidence > 0.8) return '#ef4444';
    if (confidence > 0.5) return '#f59e0b';
    return '#10b981';
  };

  return (
    <div className="history-panel">
      <div className="history-header">
        <div className="history-title">
          <Clock className="history-icon" />
          <span>Recent Analysis</span>
        </div>
      </div>
      
      <div className="history-content">
        {history.length === 0 ? (
          <div className="empty-state">
            <FileText className="empty-icon" />
            <p className="empty-text">No analysis history yet</p>
            <p className="empty-subtext">Your recent analyses will appear here</p>
          </div>
        ) : (
          <div className="history-list">
            {history.map((item) => (
              <div 
                key={item.id} 
                className="history-item"
                onClick={() => onSelectItem(item)}
              >
                <div className="history-item-content">
                  <div className="history-item-text">
                    {item.text}
                  </div>
                  <div className="history-item-meta">
                    <span className="history-time">
                      {formatTime(item.timestamp)}
                    </span>
                    {item.results && (
                      <div 
                        className="confidence-badge"
                        style={{ 
                          backgroundColor: getConfidenceColor(item.results),
                          color: 'white'
                        }}
                      >
                        {Math.round((item.results.overall_confidence || 0) * 100)}%
                      </div>
                    )}
                  </div>
                </div>
                <ChevronRight className="history-arrow" />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};