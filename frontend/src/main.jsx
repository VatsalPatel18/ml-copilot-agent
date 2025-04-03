import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx' // Your main App component
import './index.css'     // Import Tailwind CSS styles (ensure index.css exists)

// Find the root div in index.html and render the App component into it
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
