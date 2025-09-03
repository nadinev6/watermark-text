import React from 'react';
import { Search, FileText } from 'lucide-react';
import './Header.css';

interface HeaderProps {}

export const Header: React.FC<HeaderProps> = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="header-left">
          <div className="logo">
            <FileText className="logo-icon" />
            <span className="logo-text">AI Watermark Detector</span>
          </div>
        </div>
        
        <div className="header-center">
          <div className="search-bar">
            <Search className="search-icon" />
            <input 
              type="text" 
              placeholder="Search analysis history..." 
              className="search-input"
            />
          </div>
        </div>
        
        <div className="header-right">
        </div>
      </div>
    </header>
  );
};