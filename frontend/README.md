# AI Watermark Detector Frontend

A modern React TypeScript frontend for the AI Watermark Detector application.

## Features

- **Document-style Text Editor**: Clean, familiar white document interface with paper-like styling
- **Mesh Gradient Background**: Beautiful animated gradient background using CSS animations
- **Dual Panel Layout**: 
  - Left panel: Light-themed history sidebar
  - Right panel: Dark-themed results panel
- **Modern UI Elements**: Drop shadows, blur effects, and smooth animations
- **Responsive Design**: Adapts to different screen sizes
- **Real-time Analysis**: Live text analysis with confidence scoring

## Design Elements

- **Background**: Animated mesh gradient with floating orbs
- **Text Editor**: Document-style with red margin line, serif font
- **Panels**: Contrasting light/dark theme with glassmorphism effects
- **Shadows**: Layered drop shadows for depth perception
- **Colors**: Teal accent colors (#4ecdc4) with gradient highlights

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

## Components

- `App.tsx` - Main application component
- `MeshGradientBackground.tsx` - Animated gradient background
- `Header.tsx` - Top navigation with search and analyze button
- `TextEditor.tsx` - Document-style text editor
- `HistoryPanel.tsx` - Left sidebar with analysis history
- `ResultsPanel.tsx` - Right sidebar with detection results

## API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000` and uses the `/api/detect` endpoint for text analysis.