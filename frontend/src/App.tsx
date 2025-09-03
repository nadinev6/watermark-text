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
            />
          </div>
          <ResultsPanel results={results} isAnalyzing={isAnalyzing} />
        </div>
      </div>
    </div>
  );
}

export default App;