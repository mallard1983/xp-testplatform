import { useState, useEffect, useRef } from 'react'
import { getEvents } from '../api.js'

export default function ConversationFeed({ run, liveEvents, onReplayStats }) {
  const [tab, setTab] = useState('chat')
  const [allEvents, setAllEvents] = useState(null) // null = loading
  const [replayIdx, setReplayIdx] = useState(null)  // null = show all, int = show up to idx
  const feedRef = useRef(null)

  const isActive = run?.status === 'running' || run?.status === 'starting'

  // Load events for completed runs
  useEffect(() => {
    if (!run?.run_id) { setAllEvents(null); return }
    if (isActive) { setAllEvents(null); return }
    setAllEvents(null)
    setReplayIdx(null)
    if (onReplayStats) onReplayStats(null)
    getEvents(run.run_id).then(setAllEvents).catch(() => setAllEvents([]))
  }, [run?.run_id, isActive])

  // Auto-scroll during active streaming
  useEffect(() => {
    if (isActive && feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight
    }
  }, [liveEvents, isActive])

  // Report per-turn stats during replay
  useEffect(() => {
    if (!onReplayStats || !allEvents) return
    if (replayIdx === null) {
      onReplayStats(null) // back to summary view
      return
    }
    const turns = extractTurns(allEvents, false)
    const turn = turns[replayIdx]
    if (turn) onReplayStats(extractStatsFromTurn(allEvents, replayIdx))
  }, [replayIdx, allEvents])

  // Determine which events to display
  const events = isActive ? (liveEvents || []) : (allEvents || [])
  const turns = extractTurns(events, isActive)
  const visibleTurns = replayIdx === null ? turns : turns.slice(0, replayIdx + 1)

  if (!run) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div className="tabs">
        <div className={`tab ${tab === 'chat' ? 'active' : ''}`} onClick={() => setTab('chat')}>
          Chat
        </div>
        <div className={`tab ${tab === 'detail' ? 'active' : ''}`} onClick={() => setTab('detail')}>
          Detail
        </div>
        {!isActive && allEvents !== null && (
          <div className="tab" style={{ marginLeft: 'auto', cursor: 'default', color: 'var(--text2)', fontSize: 12 }}>
            {turns.length} turns
          </div>
        )}
      </div>

      {tab === 'chat' ? (
        <div className="conversation-feed" ref={feedRef}>
          {!isActive && allEvents === null && (
            <div className="text-muted text-small" style={{ padding: 16 }}>Loading…</div>
          )}
          {visibleTurns.map((turn, i) => (
            <ChatTurn key={i} turn={turn} />
          ))}
          {isActive && liveEvents && liveEvents.length === 0 && (
            <div className="text-muted text-small" style={{ padding: 16 }}>
              Starting run…
            </div>
          )}
        </div>
      ) : (
        <div className="conversation-feed" ref={feedRef}>
          {events.map((ev, i) => (
            <DetailEvent key={i} event={ev} />
          ))}
        </div>
      )}

      {/* Replay bar — completed runs only */}
      {!isActive && allEvents !== null && allEvents.length > 0 && tab === 'chat' && (
        <ReplayBar
          total={turns.length}
          current={replayIdx}
          onChange={setReplayIdx}
        />
      )}
    </div>
  )
}

function ChatTurn({ turn }) {
  return (
    <div className="turn-block">
      {turn.type === 'opening' && (
        <>
          <div className="turn-label">Opening</div>
          <div className="message interviewer">
            <div className="message-speaker">Opening Question</div>
            {turn.question}
          </div>
          <div className="message model">
            <div className="message-speaker">Model</div>
            {turn.response}
          </div>
        </>
      )}
      {turn.type === 'turn' && (
        <>
          <div className="turn-label">Turn {turn.turn}</div>
          <div className="message interviewer">
            <div className="message-speaker">Interviewer</div>
            {turn.question}
          </div>
          <div className="message model">
            <div className="message-speaker">Model</div>
            {turn.response}
          </div>
        </>
      )}
      {turn.type === 'closing' && (
        <>
          <div className="turn-label">Closing</div>
          <div className="message closing">
            <div className="message-speaker">Orchestrator</div>
            {turn.prompt}
          </div>
          <div className="message model">
            <div className="message-speaker">Model</div>
            {turn.response}
          </div>
        </>
      )}
      {turn.type === 'compaction' && (
        <div className="text-muted text-small" style={{ fontStyle: 'italic', padding: '4px 0' }}>
          ⟳ Context compacted at turn {turn.turn}
        </div>
      )}
    </div>
  )
}

function DetailEvent({ event }) {
  const { type, _ts, ...rest } = event
  return (
    <div className="detail-event">
      <div className="detail-event-type">{type || event.event}</div>
      {_ts && <div style={{ color: 'var(--text3)', marginBottom: 4, fontSize: 11 }}>{_ts}</div>}
      {JSON.stringify(rest, null, 2)}
    </div>
  )
}

function ReplayBar({ total, current, onChange }) {
  const isPlaying = current !== null
  const idx = current ?? total - 1

  return (
    <div className="replay-bar">
      <button
        className="btn btn-ghost btn-sm"
        onClick={() => onChange(isPlaying ? null : 0)}
      >
        {isPlaying ? '▶ Show All' : '↩ Replay'}
      </button>

      {isPlaying && (
        <>
          <button
            className="btn btn-ghost btn-sm"
            disabled={idx <= 0}
            onClick={() => onChange(Math.max(0, idx - 1))}
          >
            ←
          </button>

          <div
            className="replay-progress"
            onClick={e => {
              const rect = e.currentTarget.getBoundingClientRect()
              const pct = (e.clientX - rect.left) / rect.width
              onChange(Math.round(pct * (total - 1)))
            }}
          >
            <div
              className="replay-fill"
              style={{ width: `${((idx + 1) / total) * 100}%` }}
            />
          </div>

          <button
            className="btn btn-ghost btn-sm"
            disabled={idx >= total - 1}
            onClick={() => onChange(Math.min(total - 1, idx + 1))}
          >
            →
          </button>

          <span className="text-muted text-small">
            {idx + 1} / {total}
          </span>
        </>
      )}
    </div>
  )
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function extractStatsFromTurn(events, turnIdx) {
  // Find the raw event corresponding to the Nth turn in the extracted turn list
  const turnEvents = events.filter(ev => {
    const t = ev.type || ev.event
    return t === 'opening_complete' || t === 'turn_complete'
  })
  const ev = turnEvents[turnIdx]
  if (!ev) return null
  return {
    turn: ev.turn ?? 0,
    turnLimit: ev.turn_limit,
    totalTokens: ev.total_tokens,
    currentContext: ev.current_context,
    contextWindow: ev.context_window,
    activationThreshold: ev.activation_threshold,
    compactionThreshold: ev.compaction_threshold,
    compactionCount: ev.compaction_count,
    pass1Activations: ev.pass1_activations,
    pass1Tokens: ev.pass1_tokens,
    pass2Tokens: ev.pass2_tokens,
  }
}

function extractTurns(events, isLive) {
  const turns = []
  for (const ev of events) {
    const type = ev.type || ev.event
    // field names: SSE uses question/response; JSONL historically used interviewer_question/model_response
    const question = ev.question || ev.interviewer_question || ''
    const response = ev.response || ev.model_response || ''
    if (type === 'opening_complete') {
      turns.push({ type: 'opening', question, response })
    } else if (type === 'turn_complete') {
      turns.push({ type: 'turn', turn: ev.turn, question, response })
    } else if (type === 'closing' || ev.event === 'closing') {
      turns.push({ type: 'closing', prompt: ev.prompt, response: ev.response })
    } else if (type === 'compaction_event' || ev.event === 'compaction_event') {
      turns.push({ type: 'compaction', turn: ev.turn })
    }
  }
  return turns
}
