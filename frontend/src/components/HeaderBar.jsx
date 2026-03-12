import { cancelRun } from '../api.js'

export default function HeaderBar({ run, liveEvents, replayStats, onOpenConfig, onCancel }) {
  if (!run) {
    return (
      <div className="header-bar">
        <span className="header-title" style={{ color: 'var(--text2)' }}>
          XP Framework Test Platform
        </span>
        <div className="header-actions">
          <button className="btn btn-ghost btn-sm" onClick={onOpenConfig}>⚙ Configure</button>
        </div>
      </div>
    )
  }

  // Replay stats override live/summary when scrubbing through a completed run
  const stats = replayStats ? replayStats : computeStats(run, liveEvents)

  const pct = stats.turnLimit > 0
    ? Math.min(100, (stats.turn / stats.turnLimit) * 100)
    : 0

  async function handleCancel() {
    if (!run.run_id) return
    await cancelRun(run.run_id)
    onCancel()
  }

  return (
    <div className="header-bar">
      <div className="header-title">
        <span className={`badge badge-${run.condition}`} style={{ marginRight: 8 }}>
          {run.condition}
        </span>
        {run.experiment_name}
      </div>

      {/* Progress */}
      <div className="header-stat">
        <span className="header-stat-label">Turn</span>
        <span className="header-stat-value">
          {stats.turn}{stats.turnLimit ? `/${stats.turnLimit}` : ''}
        </span>
      </div>

      <div className="progress-bar" title={`${pct.toFixed(0)}% complete`}>
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>

      <div className="header-stat">
        <span className="header-stat-label">Total Tokens</span>
        <span className="header-stat-value">{formatNum(stats.totalTokens)}</span>
      </div>

      {run.condition === 'baseline' && stats.turnTokens != null && (
        <div className="header-stat">
          <span className="header-stat-label">Context</span>
          <span className="header-stat-value">{formatNum(stats.turnTokens)}</span>
        </div>
      )}

      {run.condition === 'baseline' && stats.compactionCount > 0 && (
        <div className="header-stat">
          <span className="header-stat-label">Compactions</span>
          <span className="header-stat-value">{stats.compactionCount}</span>
        </div>
      )}

      {run.condition === 'proxy' && (
        <div className="header-stat">
          <span className="header-stat-label">P1 Acts</span>
          <span className="header-stat-value">{stats.pass1Activations}</span>
        </div>
      )}
      {run.condition === 'proxy' && stats.pass1Tokens != null && (
        <div className="header-stat">
          <span className="header-stat-label">P1 Tokens</span>
          <span className="header-stat-value">{formatNum(stats.pass1Tokens)}</span>
        </div>
      )}
      {run.condition === 'proxy' && stats.pass2Tokens != null && (
        <div className="header-stat">
          <span className="header-stat-label">P2 Tokens</span>
          <span className="header-stat-value">{formatNum(stats.pass2Tokens)}</span>
        </div>
      )}

      <div className="header-stat">
        <span className="header-stat-label">Status</span>
        <span className="header-stat-value" style={{ fontSize: 12, textTransform: 'uppercase', color: statusColor(run.status) }}>
          {run.status || 'complete'}
        </span>
      </div>

      <div className="header-actions">
        {run.status === 'running' && (
          <button className="btn btn-danger btn-sm" onClick={handleCancel}>Stop</button>
        )}
        <button className="btn btn-ghost btn-sm" onClick={onOpenConfig}>⚙ Config</button>
      </div>
    </div>
  )
}

function computeStats(run, liveEvents) {
  let turn = 0
  let turnLimit = run?.parameters?.turn_limit || 0
  let totalTokens = 0
  let turnTokens = null   // context size (resets after compaction, baseline only)
  let compactionCount = 0
  let pass1Activations = 0
  let pass1Tokens = null
  let pass2Tokens = null

  if (liveEvents && liveEvents.length > 0) {
    // Live stats from streaming events
    for (const ev of liveEvents) {
      const t = ev.type
      if (t === 'opening_complete' || t === 'turn_complete') {
        turn = Math.max(turn, ev.turn ?? 0)
        turnLimit = ev.turn_limit || turnLimit
        const tok = ev.total_tokens
        if (tok) totalTokens = (tok.prompt || 0) + (tok.completion || 0)
        if (ev.turn_tokens != null) turnTokens = ev.turn_tokens
        if (ev.compaction_count != null) compactionCount = ev.compaction_count
        if (ev.pass1_activations != null) pass1Activations = ev.pass1_activations
        if (ev.pass1_tokens) pass1Tokens = (ev.pass1_tokens.prompt || 0) + (ev.pass1_tokens.completion || 0)
        if (ev.pass2_tokens) pass2Tokens = (ev.pass2_tokens.prompt || 0) + (ev.pass2_tokens.completion || 0)
      }
    }
  } else if (run?.turns_completed != null) {
    // Completed run — read from summary (no per-turn detail)
    turn = run.turns_completed
    turnLimit = run.turn_limit || turnLimit
    const t = run.total_tokens
    if (t) totalTokens = (t.prompt || 0) + (t.completion || 0)
    compactionCount = (run.compaction_events || []).length
    pass1Activations = run.pass1_activations || 0
    if (run.pass1_tokens) pass1Tokens = (run.pass1_tokens.prompt || 0) + (run.pass1_tokens.completion || 0)
    if (run.pass2_tokens) pass2Tokens = (run.pass2_tokens.prompt || 0) + (run.pass2_tokens.completion || 0)
  }

  return { turn, turnLimit, totalTokens, turnTokens, compactionCount, pass1Activations, pass1Tokens, pass2Tokens }
}

function formatNum(n) {
  if (!n) return '0'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return String(n)
}

function statusColor(status) {
  const map = {
    running: 'var(--accent)',
    complete: 'var(--success)',
    error: 'var(--error)',
    cancelled: 'var(--text3)',
    starting: 'var(--warning)',
  }
  return map[status] || 'var(--text2)'
}
