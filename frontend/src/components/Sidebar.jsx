import { useState } from 'react'
import { startRun, cancelRun, streamRun } from '../api.js'

export default function Sidebar({
  runs,
  experiments,
  selectedRunId,
  onSelectRun,
  onOpenConfig,
  onRunStarted,
  onEventReceived,
  onRefresh,
}) {
  const [launching, setLaunching] = useState(false)
  const [startDialog, setStartDialog] = useState(null) // {experiment}

  // Group runs by experiment name
  const runsByExp = {}
  for (const run of runs) {
    const key = run.experiment_name || run.experiment_id
    if (!runsByExp[key]) runsByExp[key] = []
    runsByExp[key].push(run)
  }

  async function handleStart(experiment, condition, dbSource = 'new') {
    setLaunching(true)
    try {
      const result = await startRun({
        experiment_id: experiment.id,
        condition,
        db_source: dbSource,
      })
      const runId = result.run_id
      onRunStarted(runId)

      // Connect SSE stream
      streamRun(
        runId,
        (event) => onEventReceived(runId, event),
        () => onRefresh(),
        () => onRefresh(),
      )
    } catch (err) {
      alert(`Failed to start run: ${err.message}`)
    } finally {
      setLaunching(false)
      setStartDialog(null)
    }
  }

  async function handleCancel(runId, e) {
    e.stopPropagation()
    await cancelRun(runId)
    onRefresh()
  }

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <span>Runs</span>
        <button className="btn btn-ghost btn-sm" onClick={onOpenConfig}>
          ⚙ Config
        </button>
      </div>

      <div className="sidebar-body">
        {experiments.map(exp => (
          <div key={exp.id}>
            <div className="sidebar-section">{exp.name}</div>

            {/* Start buttons */}
            <div style={{ padding: '4px 12px 8px', display: 'flex', gap: 6 }}>
              <button
                className="btn btn-ghost btn-sm flex-1"
                disabled={launching}
                onClick={() => handleStart(exp, 'baseline')}
              >
                + Baseline
              </button>
              <button
                className="btn btn-ghost btn-sm flex-1"
                disabled={launching}
                onClick={() => setStartDialog({ experiment: exp })}
              >
                + Proxy
              </button>
            </div>

            {/* Existing runs for this experiment */}
            {(runsByExp[exp.name] || []).map(run => (
              <RunItem
                key={run.run_id}
                run={run}
                selected={run.run_id === selectedRunId}
                onSelect={() => onSelectRun(run.run_id)}
                onCancel={handleCancel}
              />
            ))}
          </div>
        ))}

        {experiments.length === 0 && (
          <div className="empty-state" style={{ padding: 24, fontSize: 12 }}>
            <div className="text-muted">No experiments configured.</div>
            <button className="btn btn-ghost btn-sm" onClick={onOpenConfig}>
              Open Config
            </button>
          </div>
        )}
      </div>

      {/* Proxy start dialog — choose db source */}
      {startDialog && (
        <ProxyStartDialog
          experiment={startDialog.experiment}
          runs={runs.filter(r => r.condition === 'proxy' && r.experiment_id === startDialog.experiment.id)}
          onStart={(dbSource) => handleStart(startDialog.experiment, 'proxy', dbSource)}
          onClose={() => setStartDialog(null)}
        />
      )}
    </div>
  )
}

function RunItem({ run, selected, onSelect, onCancel }) {
  const statusClass = run.status || 'complete'
  const date = run.timestamp
    ? run.timestamp.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, '$1-$2-$3 $4:$5')
    : ''

  return (
    <div
      className={`run-item ${selected ? 'selected' : ''}`}
      onClick={onSelect}
    >
      <div className="run-item-name">
        <span className="status-dot" style={{ verticalAlign: 'middle' }}
          title={run.status}
          data-status={statusClass}
        />
        <StatusDot status={statusClass} />
        <span className={`badge badge-${run.condition}`}>{run.condition}</span>
        {' '}
        <span style={{ fontSize: 11, color: 'var(--text2)' }}>{date}</span>
      </div>
      {run.status === 'running' && (
        <button
          className="btn btn-danger btn-sm"
          style={{ marginTop: 4 }}
          onClick={(e) => onCancel(run.run_id, e)}
        >
          Stop
        </button>
      )}
    </div>
  )
}

function StatusDot({ status }) {
  return <span className={`status-dot ${status}`} style={{ marginRight: 6 }} />
}

function ProxyStartDialog({ experiment, runs, onStart, onClose }) {
  const [dbSource, setDbSource] = useState('new')

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog" onClick={e => e.stopPropagation()}>
        <div className="dialog-title">Start Proxy Run — {experiment.name}</div>
        <div className="dialog-body">
          <div className="form-group">
            <label className="form-label">Substrate database</label>
            <select
              className="form-select"
              value={dbSource}
              onChange={e => setDbSource(e.target.value)}
            >
              <option value="new">New (clean substrate)</option>
              {runs.map(r => (
                <option key={r.run_id} value={r.timestamp}>
                  From run: {r.timestamp}
                </option>
              ))}
            </select>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text2)' }}>
            {dbSource === 'new'
              ? 'Start with a fresh, empty substrate.'
              : "Copy a prior run's substrate \u2014 the source is never modified."}
          </div>
        </div>
        <div className="dialog-actions">
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={() => onStart(dbSource)}>Start</button>
        </div>
      </div>
    </div>
  )
}
