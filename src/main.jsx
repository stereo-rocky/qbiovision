// ================================================================
// main.jsx — React 18 entry point for Q-BioVision
// ================================================================
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './styles/globals.css'

// Mount the React app into the #root div defined in index.html
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
