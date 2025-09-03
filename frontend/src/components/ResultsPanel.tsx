import React from 'react';
import { 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  BarChart3, 
  Brain,
  Zap,
  Eye,
  Shield,
  ShieldCheck,
  ShieldX,
  Info,
  Clock,
  Hash,
  FileText
} from 'lucide-react';
import './ResultsPanel.css';

interface ResultsPanelProps {
  results: any;
  isAnalyzing: boolean;
  watermarkInfo?: {
    hasWatermark: boolean;
    extractedContent?: string;
    confidence?: number;
    method?: string;
  };
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({ results, isAnalyzing, watermarkInfo }) => {
  const getOverallStatus = () => {
    if (!results) return null;
    
    const confidence = results.overall_confidence || 0;
    if (confidence > 0.8) {
      return {
        icon: XCircle,
        text: 'AI Generated',
        color: '#ef4444',
        bgColor: 'rgba(239, 68, 68, 0.1)'
      };
    } else if (confidence > 0.5) {
      return {
        icon: AlertTriangle,
        text: 'Possibly AI',
        color: '#f59e0b',
        bgColor: 'rgba(245, 158, 11, 0.1)'
      };
    } else {
      return {
        icon: CheckCircle,
        text: 'Likely Human',
        color: '#10b981',
        bgColor: 'rgba(16, 185, 129, 0.1)'
      };
    }
  };

  const renderDetectorResult = (detector: any, index: number) => {
    const confidence = detector.confidence || 0;
    const isPositive = confidence > 0.5;
    
    return (
      <div key={index} className="detector-result">
        <div className="detector-header">
          <div className="detector-info">
            <Brain className="detector-icon" />
            <span className="detector-name">{detector.name}</span>
          </div>
          <div className={`detector-status ${isPositive ? 'positive' : 'negative'}`}>
            {isPositive ? 'AI Detected' : 'Human-like'}
          </div>
        </div>
        
        <div className="confidence-bar">
          <div className="confidence-label">
            <span>Confidence</span>
            <span className="confidence-value">{Math.round(confidence * 100)}%</span>
          </div>
          <div className="confidence-track">
            <div 
              className="confidence-fill"
              style={{ 
                width: `${confidence * 100}%`,
                backgroundColor: isPositive ? '#ef4444' : '#10b981'
              }}
            />
          </div>
        </div>
        
        {detector.details && (
          <div className="detector-details">
            <div className="detail-item">
              <Eye className="detail-icon" />
              <span>{detector.details}</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderWatermarkSection = () => {
    if (!watermarkInfo) return null;

    const { hasWatermark, extractedContent, confidence, method } = watermarkInfo;

    return (
      <div className="watermark-section">
        <h4 className="section-title">Watermark Information</h4>
        
        <div className="watermark-status-card">
          <div className="watermark-header">
            <div className="watermark-icon-container">
              {hasWatermark ? (
                <ShieldCheck className="watermark-icon watermark-found" />
              ) : (
                <ShieldX className="watermark-icon watermark-not-found" />
              )}
            </div>
            <div className="watermark-info">
              <h3 className="watermark-status-text">
                {hasWatermark ? 'Watermark Detected' : 'No Watermark Found'}
              </h3>
              {confidence !== undefined && (
                <p className="watermark-confidence">
                  Confidence: {Math.round(confidence * 100)}%
                </p>
              )}
            </div>
          </div>

          {hasWatermark && extractedContent && (
            <div className="watermark-details">
              <div className="watermark-content">
                <div className="detail-label">
                  <FileText className="detail-icon" />
                  <span>Extracted Content</span>
                </div>
                <div className="watermark-text">
                  "{extractedContent}"
                </div>
              </div>
              
              {method && (
                <div className="watermark-method">
                  <div className="detail-label">
                    <Info className="detail-icon" />
                    <span>Detection Method</span>
                  </div>
                  <div className="method-badge">
                    {method === 'stegano_lsb' ? 'Steganographic LSB' : 'Visible Text'}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="results-panel">
      <div className="results-header">
        <div className="results-title">
          <BarChart3 className="results-icon" />
          <span>Analysis Results</span>
        </div>
      </div>
      
      <div className="results-content">
        {isAnalyzing ? (
          <div className="analyzing-state">
            <div className="analyzing-spinner">
              <Zap className="spinner-icon" />
            </div>
            <h3 className="analyzing-title">Analyzing Text</h3>
            <p className="analyzing-text">
              Running advanced AI detection algorithms...
            </p>
            <div className="analyzing-steps">
              <div className="step active">Preprocessing text</div>
              <div className="step active">Running detectors</div>
              <div className="step">Calculating confidence</div>
              <div className="step">Generating report</div>
            </div>
          </div>
        ) : !results ? (
          <div className="empty-results">
            <h3 className="empty-title">Ready to Analyze</h3>
            <p className="empty-text">
              Enter text in the editor and click "Analyze Text" to detect AI-generated content.
            </p>
            <div className="features-list">
              <div className="feature-item">
                <CheckCircle className="feature-icon" />
                <span>Multiple AI detection algorithms</span>
              </div>
              <div className="feature-item">
                <CheckCircle className="feature-icon" />
                <span>Confidence scoring</span>
              </div>
              <div className="feature-item">
                <CheckCircle className="feature-icon" />
                <span>Detailed analysis breakdown</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="results-data">
            {renderWatermarkSection()}
            
            <div className="overall-result">
              {(() => {
                const status = getOverallStatus();
                if (!status) return null;
                
                const Icon = status.icon;
                return (
                  <div 
                    className="overall-card"
                    style={{ 
                      backgroundColor: status.bgColor,
                      borderColor: status.color + '40'
                    }}
                  >
                    <div className="overall-header">
                      <Icon 
                        className="overall-icon" 
                        style={{ color: status.color }}
                      />
                      <div className="overall-info">
                        <h3 className="overall-status" style={{ color: status.color }}>
                          {status.text}
                        </h3>
                        <p className="overall-confidence">
                          {Math.round((results.overall_confidence || 0) * 100)}% confidence
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
            
            <div className="detectors-section">
              <h4 className="section-title">Detector Results</h4>
              <div className="detectors-list">
                {results.detectors?.map((detector: any, index: number) => 
                  renderDetectorResult(detector, index)
                )}
              </div>
            </div>
            
            {results.metadata && (
              <div className="metadata-section">
                <h4 className="section-title">Analysis Metadata</h4>
                <div className="metadata-grid">
                  <div className="metadata-item">
                    <span className="metadata-label">Processing Time</span>
                    <span className="metadata-value">
                      {results.metadata.processing_time || 'N/A'}
                    </span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Text Length</span>
                    <span className="metadata-value">
                      {results.metadata.text_length || 'N/A'} chars
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};