import { useState, useEffect, useCallback } from 'react'
import { getRuns, getExperiments } from './api.js'
import Sidebar from './components/Sidebar.jsx'
import HeaderBar from './components/HeaderBar.jsx'
import ConversationFeed from './components/ConversationFeed.jsx'
import ConfigPanel from './components/ConfigPanel.jsx'

export default function App() {
  const [runs, setRuns] = useState([])
  const [experiments, setExperiments] = useState([])
  const [selectedRunId, setSelectedRunId] = useState(null)
  const [configOpen, setConfigOpen] = useState(false)
  const [activeEvents, setActiveEvents] = useState({}) // run_id → [events]
  const [replayStats, setReplayStats] = useState(null) // stats at current replay turn

  // ── Data loading ───────────────────────────────────────────────────────────

  const refresh = useCallback(async () => {
    try {
      const [r, e] = await Promise.all([getRuns(), getExperiments()])
      setRuns(r)
      setExperiments(e)
    } catch (err) {
      console.error('refresh failed', err)
    }
  }, [])

  useEffect(() => {
    refresh()
    const interval = setInterval(refresh, 5000)
    return () => clearInterval(interval)
  }, [refresh])

  // ── Event accumulation for streaming runs ──────────────────────────────────

  const appendEvent = useCallback((runId, event) => {
    setActiveEvents(prev => ({
      ...prev,
      [runId]: [...(prev[runId] || []), event],
    }))
  }, [])

  const onRunStarted = useCallback((runId) => {
    setActiveEvents(prev => ({ ...prev, [runId]: [] }))
    setSelectedRunId(runId)
    setReplayStats(null)
    refresh()
  }, [refresh])

  // ── Selected run data ──────────────────────────────────────────────────────

  const selectedRun = runs.find(r => r.run_id === selectedRunId) || null
  const liveEvents = activeEvents[selectedRunId] || null

  // Clear replay stats when switching runs
  const handleSelectRun = useCallback((runId) => {
    setSelectedRunId(runId)
    setReplayStats(null)
  }, [])

  return (
    <div className="layout">
      <Sidebar
        runs={runs}
        experiments={experiments}
        selectedRunId={selectedRunId}
        onSelectRun={handleSelectRun}
        onOpenConfig={() => setConfigOpen(true)}
        onRunStarted={onRunStarted}
        onEventReceived={appendEvent}
        onRefresh={refresh}
      />

      <div className="main">
        <HeaderBar
          run={selectedRun}
          liveEvents={liveEvents}
          replayStats={replayStats}
          onOpenConfig={() => setConfigOpen(true)}
          onCancel={refresh}
        />

        <div className="content">
          {selectedRunId ? (
            <ConversationFeed
              run={selectedRun}
              liveEvents={liveEvents}
              onReplayStats={setReplayStats}
            />
          ) : (
            <div className="empty-state">
              <div className="empty-state-title">XP Framework Test Platform</div>
              <div className="text-muted">Select a run from the sidebar, or start a new one.</div>
            </div>
          )}
        </div>
      </div>

      {configOpen && (
        <ConfigPanel
          experiments={experiments}
          onClose={() => { setConfigOpen(false); refresh() }}
        />
      )}
    </div>
  )
}
