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
  const [startDialog, setStartDialog] = useState(null) // {experiment, mode} mode: 'proxy'|'both'

  // Group runs by experiment name
  const runsByExp = {}
  for (const run of runs) {
    const key = run.experiment_name || run.experiment_id
    if (!runsByExp[key]) runsByExp[key] = []
    runsByExp[key].push(run)
  }

  const queuedRuns = runs.filter(r => r.status === 'queued')

  async function enqueue(experiment, condition, dbSource = 'new') {
    const result = await startRun({
      experiment_id: experiment.id,
      condition,
      db_source: dbSource,
    })
    const runId = result.run_id
    onRunStarted(runId)
    // Connect SSE now — it will sit waiting until the run is dequeued and starts
    streamRun(
      runId,
      (event) => onEventReceived(runId, event),
      () => onRefresh(),
      () => onRefresh(),
    )
    return runId
  }

  async function handleBaseline(experiment) {
    setLaunching(true)
    try {
      await enqueue(experiment, 'baseline')
    } catch (err) {
      alert(`Failed to queue run: ${err.message}`)
    } finally {
      setLaunching(false)
    }
  }

  async function handleProxyOrBoth(experiment, mode, dbSource) {
    setLaunching(true)
    try {
      if (mode === 'both') {
        await enqueue(experiment, 'baseline')
      }
      await enqueue(experiment, 'proxy', dbSource)
    } catch (err) {
      alert(`Failed to queue run: ${err.message}`)
    } finally {
      setLaunching(false)
      setStartDialog(null)
    }
  }

  async function handleDequeue(runId, e) {
    e.stopPropagation()
    await cancelRun(runId)
    onRefresh()
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
                onClick={() => handleBaseline(exp)}
              >
                + Baseline
              </button>
              <button
                className="btn btn-ghost btn-sm flex-1"
                disabled={launching}
                onClick={() => setStartDialog({ experiment: exp, mode: 'proxy' })}
              >
                + Proxy
              </button>
              <button
                className="btn btn-ghost btn-sm flex-1"
                disabled={launching}
                onClick={() => setStartDialog({ experiment: exp, mode: 'both' })}
              >
                + Both
              </button>
            </div>

            {/* Runs for this experiment */}
            {(runsByExp[exp.name] || []).map(run => (
              <RunItem
                key={run.run_id}
                run={run}
                selected={run.run_id === selectedRunId}
                onSelect={() => onSelectRun(run.run_id)}
                onCancel={handleCancel}
                onDequeue={handleDequeue}
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

      {/* Queue panel */}
      {queuedRuns.length > 0 && (
        <div className="queue-panel">
          <div className="queue-panel-header">
            <span>Queue — {queuedRuns.length} pending</span>
            <button
              className="btn btn-ghost btn-sm"
              onClick={async () => {
                for (const r of queuedRuns) await cancelRun(r.run_id)
                onRefresh()
              }}
            >
              Clear
            </button>
          </div>
          {queuedRuns.map(r => (
            <div key={r.run_id} className="queue-item">
              <span className={`badge badge-${r.condition}`}>{r.condition}</span>
              <span className="queue-item-name">{r.experiment_name}</span>
              <button
                className="btn btn-ghost btn-sm"
                style={{ marginLeft: 'auto', fontSize: 11 }}
                onClick={(e) => handleDequeue(r.run_id, e)}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Proxy / Both start dialog */}
      {startDialog && (
        <ProxyStartDialog
          experiment={startDialog.experiment}
          mode={startDialog.mode}
          runs={runs.filter(r => r.condition === 'proxy' && r.experiment_id === startDialog.experiment.id)}
          onStart={(dbSource) => handleProxyOrBoth(startDialog.experiment, startDialog.mode, dbSource)}
          onClose={() => setStartDialog(null)}
        />
      )}
    </div>
  )
}

function RunItem({ run, selected, onSelect, onCancel, onDequeue }) {
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
      {run.status === 'queued' && (
        <button
          className="btn btn-ghost btn-sm"
          style={{ marginTop: 4, fontSize: 11 }}
          onClick={(e) => onDequeue(run.run_id, e)}
        >
          Remove
        </button>
      )}
    </div>
  )
}

function StatusDot({ status }) {
  return <span className={`status-dot ${status}`} style={{ marginRight: 6 }} />
}

function ProxyStartDialog({ experiment, mode, runs, onStart, onClose }) {
  const [dbSource, setDbSource] = useState('new')
  const isProxy = mode === 'proxy'

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog" onClick={e => e.stopPropagation()}>
        <div className="dialog-title">
          {isProxy ? 'Queue Proxy Run' : 'Queue Both Runs'} — {experiment.name}
        </div>
        <div className="dialog-body">
          {!isProxy && (
            <p className="text-muted text-small" style={{ marginBottom: 12 }}>
              Baseline will be queued first, then Proxy.
            </p>
          )}
          <div className="form-group">
            <label className="form-label">Proxy substrate database</label>
            <select
              className="form-select"
              value={dbSource}
              onChange={e => setDbSource(e.target.value)}
            >
              <option value="new">New (clean substrate)</option>
              {runs.filter(r => r.status === 'complete').map(r => (
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
          <button className="btn btn-primary" onClick={() => onStart(dbSource)}>
            {isProxy ? 'Queue Proxy' : 'Queue Both'}
          </button>
        </div>
      </div>
    </div>
  )
}
