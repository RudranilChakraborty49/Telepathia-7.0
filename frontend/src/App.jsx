import { useState } from 'react'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Backtest from './pages/Backtest'

export default function App() {
  // 'landing' | 'dashboard' | 'backtest'
  const [view, setView] = useState('landing')

  return (
    <>
      {view === 'landing' && (
        <Landing onEnter={() => setView('dashboard')} />
      )}
      {view === 'dashboard' && (
        <Dashboard
          onBack={() => setView('landing')}
          onShowBacktest={() => setView('backtest')}
        />
      )}
      {view === 'backtest' && (
        <Backtest onBack={() => setView('dashboard')} />
      )}
    </>
  )
}
