import React from 'react';
import { MeshGradientBackground } from './components/MeshGradientBackground';
import { TextEditor } from './components/TextEditor';
import { ResultsPanel } from './components/ResultsPanel';
import { HistoryPanel } from './components/HistoryPanel';
import { Header } from './components/Header';
import './App.css';

interface HistoryItem {
  id: number;
  text: string;
  timestamp: string;
  results: any;
}

function App() {
  const [text, setText] = React.useState('');
  const [results, setResults] = React.useState<any>(null);
  const [history, setHistory] = React.useState<HistoryItem[]>([]);
  const [isAnalyzing, setIsAnalyzing] = React.useState(false);
  const [watermarkInfo, setWatermarkInfo] = React.useState<{
    hasWatermark: boolean;
    extractedContent?: string;
    confidence?: number;
    method?: string;
  }>({ hasWatermark: false });

  // Auto-extract watermark when text changes
  React.useEffect(() => {
    const extractWatermark = async () => {
      if (!text.trim()) {
        setWatermarkInfo({ hasWatermark: false });
        return;
      }

      try {
        const response = await fetch('/api/watermark/extract', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: text,
            methods: ['stegano_lsb', 'visible_text']
          }),
        });

        if (response.ok) {
          const result = await response.json();
          setWatermarkInfo({
            hasWatermark: result.watermark_found,
            extractedContent: result.watermark_content,
            confidence: result.confidence_score,
            method: result.extraction_method
          });
        } else {
          setWatermarkInfo({ hasWatermark: false });
        }
      } catch (error) {
        console.error('Auto watermark extraction failed:', error);
        setWatermarkInfo({ hasWatermark: false });
      }
    };

    // Debounce the extraction to avoid too many API calls
    const timeoutId = setTimeout(extractWatermark, 1000);
    return () => clearTimeout(timeoutId);
  }, [text]);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    
    setIsAnalyzing(true);
    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      const result = await response.json();
      setResults(result);
      
      // Add to history
      const historyItem = {
        id: Date.now(),
        text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
        timestamp: new Date().toISOString(),
        results: result,
      };
      setHistory(prev => [historyItem, ...prev.slice(0, 9)]); // Keep last 10
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app">
      <MeshGradientBackground />
      <div className="app-content">
        <Header />
        <div className="main-layout">
          <HistoryPanel history={history} onSelectItem={(item) => setText(item.text)} />
          <div className="editor-container">
            <TextEditor 
              value={text} 
              onChange={setText}
              placeholder="Paste or type your text here to analyze for AI watermarks..."
              onAnalyze={handleAnalyze}
              isAnalyzing={isAnalyzing}
              watermarkInfo={watermarkInfo}
              onWatermarkChange={setWatermarkInfo}
            />
          </div>
          <ResultsPanel 
            results={results} 
            isAnalyzing={isAnalyzing} 
            watermarkInfo={watermarkInfo}
          />
        </div>
      </div>
    </div>
  );
}

export default App;