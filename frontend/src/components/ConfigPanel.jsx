import { useState, useEffect } from 'react'
import {
  getModels, createModel, updateModel, deleteModel, setModelKey,
  getDefaultPrompts, getPrompts, createPrompt, updatePrompt, deletePrompt,
  getMcpServers, createMcpServer, updateMcpServer, deleteMcpServer,
  getExperiments, createExperiment, updateExperiment, deleteExperiment,
  getGlobalConfig, updateGlobalConfig,
  getSearchConfig, updateSearchConfig,
} from '../api.js'

const TABS = ['Globals', 'Models', 'Prompts', 'MCP', 'Tests']

export default function ConfigPanel({ experiments: initialExperiments, onClose }) {
  const [tab, setTab] = useState('Globals')

  return (
    <div className="overlay" onClick={onClose}>
      <div className="config-panel" onClick={e => e.stopPropagation()}>
        <div className="config-panel-header">
          Configuration
          <button className="btn btn-ghost btn-sm" onClick={onClose}>✕ Close</button>
        </div>
        <div className="config-panel-tabs">
          {TABS.map(t => (
            <div
              key={t}
              className={`tab ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
            >
              {t}
            </div>
          ))}
        </div>
        <div className="config-panel-body">
          {tab === 'Globals' && <GlobalsTab />}
          {tab === 'Models'  && <ModelsTab />}
          {tab === 'Prompts' && <PromptsTab />}
          {tab === 'MCP'     && <McpTab />}
          {tab === 'Tests'   && <TestsTab />}
        </div>
      </div>
    </div>
  )
}

// ── Globals ────────────────────────────────────────────────────────────────────

function GlobalsTab() {
  const [cfg, setCfg] = useState(null)
  const [search, setSearch] = useState(null)
  const [saving, setSaving] = useState(false)
  const [keyInput, setKeyInput] = useState('')
  const [showKey, setShowKey] = useState(false)

  useEffect(() => {
    Promise.all([getGlobalConfig(), getSearchConfig()]).then(([g, s]) => {
      setCfg(g)
      setSearch(s)
    })
  }, [])

  async function saveGlobals() {
    setSaving(true)
    try { await updateGlobalConfig(cfg) } finally { setSaving(false) }
  }

  async function saveSearch() {
    setSaving(true)
    try {
      const payload = { provider: search.provider, enabled: search.enabled }
      if (keyInput) payload.api_key = keyInput
      await updateSearchConfig(payload)
      setKeyInput('')
    } finally { setSaving(false) }
  }

  if (!cfg || !search) return <div className="text-muted">Loading…</div>

  return (
    <div>
      <h3 style={{ marginBottom: 16 }}>Experiment Defaults</h3>
      <div className="form-row">
        <Field label="Turn Limit" value={cfg.turn_limit} onChange={v => setCfg({...cfg, turn_limit: +v})} type="number" />
        <Field label="Context Window" value={cfg.context_window} onChange={v => setCfg({...cfg, context_window: +v})} type="number" />
      </div>
      <div className="form-row">
        <Field label="Compaction Threshold" value={cfg.compaction_threshold_fraction} onChange={v => setCfg({...cfg, compaction_threshold_fraction: +v})} type="number" step="0.01" />
        <Field label="Pass 1 Activation" value={cfg.pass1_activation_fraction} onChange={v => setCfg({...cfg, pass1_activation_fraction: +v})} type="number" step="0.01" />
      </div>
      <div className="form-row">
        <Field label="Turn Pause Min (seconds)" value={cfg.turn_pause_min_seconds} onChange={v => setCfg({...cfg, turn_pause_min_seconds: +v})} type="number" step="1" />
        <Field label="Turn Pause Max (seconds)" value={cfg.turn_pause_max_seconds} onChange={v => setCfg({...cfg, turn_pause_max_seconds: +v})} type="number" step="1" />
      </div>
      <button className="btn btn-primary btn-sm" disabled={saving} onClick={saveGlobals} style={{ marginBottom: 24 }}>
        Save Defaults
      </button>

      <div className="divider" />
      <h3 style={{ marginBottom: 16 }}>Search</h3>
      <div className="form-row">
        <Field label="Provider" value={search.provider} onChange={v => setSearch({...search, provider: v})} />
        <div className="form-group">
          <label className="form-label">Enabled</label>
          <select className="form-select" value={search.enabled ? 'yes' : 'no'} onChange={e => setSearch({...search, enabled: e.target.value === 'yes'})}>
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </div>
      </div>
      <div className="form-group">
        <label className="form-label">
          API Key {search.key_configured ? '(configured ✓)' : '(not set)'}
        </label>
        <div className="flex gap-8">
          <input
            className="form-input flex-1"
            type={showKey ? 'text' : 'password'}
            placeholder="Enter new key to update…"
            value={keyInput}
            onChange={e => setKeyInput(e.target.value)}
          />
          <button className="btn btn-ghost btn-sm" onClick={() => setShowKey(!showKey)}>
            {showKey ? 'Hide' : 'Show'}
          </button>
        </div>
      </div>
      <button className="btn btn-primary btn-sm" disabled={saving} onClick={saveSearch}>
        Save Search Config
      </button>
    </div>
  )
}

// ── Models ─────────────────────────────────────────────────────────────────────

function ModelsTab() {
  const [models, setModels] = useState([])
  const [adding, setAdding] = useState(false)
  const [editing, setEditing] = useState(null)
  const [keyModal, setKeyModal] = useState(null)

  const load = () => getModels().then(setModels)
  useEffect(() => { load() }, [])

  async function handleCreate(data) {
    await createModel(data)
    setAdding(false)
    load()
  }

  async function handleUpdate(id, data) {
    await updateModel(id, data)
    setEditing(null)
    load()
  }

  async function handleDelete(id) {
    if (!confirm('Delete this model?')) return
    await deleteModel(id)
    load()
  }

  async function handleSetKey(id, key) {
    await setModelKey(id, key)
    setKeyModal(null)
  }

  return (
    <div>
      <div className="flex items-center gap-8 mb-16">
        <h3 style={{ flex: 1 }}>Model Store</h3>
        <button className="btn btn-primary btn-sm" onClick={() => setAdding(!adding)}>
          {adding ? 'Cancel' : '+ Add Model'}
        </button>
      </div>

      {adding && (
        <ModelForm onSave={handleCreate} onCancel={() => setAdding(false)} />
      )}

      <div className="store-list">
        {models.map(m => (
          <div key={m.id} className="store-card">
            <div className="store-card-info">
              <div className="store-card-name">{m.name}</div>
              <div className="store-card-meta">{m.model_identifier} · {m.endpoint_url} · {m.context_window.toLocaleString()} ctx</div>
            </div>
            <div className="store-card-actions">
              <button className="btn btn-ghost btn-sm" onClick={() => setKeyModal(m.id)}>Key</button>
              <button className="btn btn-ghost btn-sm" onClick={() => setEditing(m.id)}>Edit</button>
              <button className="btn btn-danger btn-sm" onClick={() => handleDelete(m.id)}>Del</button>
            </div>
          </div>
        ))}
        {models.length === 0 && !adding && <div className="text-muted text-small">No models configured.</div>}
      </div>

      {editing && (
        <div className="overlay" onClick={() => setEditing(null)}>
          <div className="dialog" onClick={e => e.stopPropagation()}>
            <ModelForm
              model={models.find(m => m.id === editing)}
              onSave={d => handleUpdate(editing, d)}
              onCancel={() => setEditing(null)}
            />
          </div>
        </div>
      )}

      {keyModal && (
        <KeyDialog
          label="Model API Key"
          onSave={key => handleSetKey(keyModal, key)}
          onClose={() => setKeyModal(null)}
        />
      )}
    </div>
  )
}

function ModelForm({ model, onSave, onCancel }) {
  const [name, setName] = useState(model?.name || '')
  const [identifier, setIdentifier] = useState(model?.model_identifier || '')
  const [endpoint, setEndpoint] = useState(model?.endpoint_url || 'https://api.ollama.com')
  const [ctxWindow, setCtxWindow] = useState(model?.context_window || 256000)

  function save() {
    if (!name.trim() || !identifier.trim() || !endpoint.trim()) return
    onSave({ name: name.trim(), model_identifier: identifier.trim(), endpoint_url: endpoint.trim(), context_window: +ctxWindow })
  }

  return (
    <div className="inline-add">
      <div className="inline-add-title">{model ? 'Edit Model' : 'New Model'}</div>
      <Field label="Display Name" value={name} onChange={setName} placeholder="e.g. Qwen 3.5 Cloud" />
      <Field label="Model Identifier" value={identifier} onChange={setIdentifier} placeholder="e.g. qwen3.5:cloud" />
      <Field label="Endpoint URL" value={endpoint} onChange={setEndpoint} placeholder="https://api.ollama.com" />
      <Field label="Context Window (tokens)" value={ctxWindow} onChange={v => setCtxWindow(+v)} type="number" />
      <div className="flex gap-8">
        <button className="btn btn-primary btn-sm" onClick={save}>Save</button>
        <button className="btn btn-ghost btn-sm" onClick={onCancel}>Cancel</button>
      </div>
    </div>
  )
}

// ── Prompts ────────────────────────────────────────────────────────────────────

function PromptsTab() {
  const [defaults, setDefaults] = useState([])
  const [prompts, setPrompts] = useState([])
  const [adding, setAdding] = useState(false)
  const [editing, setEditing] = useState(null)
  const [expanding, setExpanding] = useState(null) // key of expanded default

  const load = () => Promise.all([getDefaultPrompts(), getPrompts()]).then(([d, p]) => {
    setDefaults(d); setPrompts(p)
  })
  useEffect(() => { load() }, [])

  async function handleCreate(data) { await createPrompt(data); setAdding(false); load() }
  async function handleUpdate(id, data) { await updatePrompt(id, data); setEditing(null); load() }
  async function handleDelete(id) { if (!confirm('Delete this prompt?')) return; await deletePrompt(id); load() }

  return (
    <div>
      <h3 style={{ marginBottom: 8 }}>Built-in Defaults</h3>
      <p className="text-muted text-small" style={{ marginBottom: 12 }}>
        Read-only. Add a prompt to the store to override any of these for a specific test.
      </p>
      <div className="store-list" style={{ marginBottom: 24 }}>
        {defaults.map(d => (
          <div key={d.key} className="store-card" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div className="store-card-name">{d.name}</div>
                <div className="store-card-meta" style={{ fontFamily: 'monospace', fontSize: 11 }}>{d.key}.txt</div>
              </div>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setExpanding(expanding === d.key ? null : d.key)}
              >
                {expanding === d.key ? 'Collapse' : 'View'}
              </button>
            </div>
            {expanding === d.key && (
              <pre style={{
                background: 'var(--bg)', padding: 12, borderRadius: 4, fontSize: 12,
                whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: 'var(--text2)',
                maxHeight: 300, overflowY: 'auto', margin: 0,
              }}>
                {d.content}
              </pre>
            )}
          </div>
        ))}
      </div>

      <div className="flex items-center gap-8 mb-16">
        <h3 style={{ flex: 1 }}>Store Overrides</h3>
        <button className="btn btn-primary btn-sm" onClick={() => setAdding(!adding)}>
          {adding ? 'Cancel' : '+ Add Prompt'}
        </button>
      </div>

      {adding && <PromptForm onSave={handleCreate} onCancel={() => setAdding(false)} />}

      <div className="store-list">
        {prompts.map(p => (
          <div key={p.id} className="store-card">
            <div className="store-card-info">
              <div className="store-card-name">{p.name}</div>
              <div className="store-card-meta">{p.content.slice(0, 80)}…</div>
            </div>
            <div className="store-card-actions">
              <button className="btn btn-ghost btn-sm" onClick={() => setEditing(p.id)}>Edit</button>
              <button className="btn btn-danger btn-sm" onClick={() => handleDelete(p.id)}>Del</button>
            </div>
          </div>
        ))}
        {prompts.length === 0 && !adding && (
          <div className="text-muted text-small">No overrides — built-in defaults are used for all tests.</div>
        )}
      </div>

      {editing && (
        <div className="overlay" onClick={() => setEditing(null)}>
          <div className="dialog" style={{ width: 'min(640px,90vw)' }} onClick={e => e.stopPropagation()}>
            <PromptForm
              prompt={prompts.find(p => p.id === editing)}
              onSave={d => handleUpdate(editing, d)}
              onCancel={() => setEditing(null)}
            />
          </div>
        </div>
      )}
    </div>
  )
}

function PromptForm({ prompt, onSave, onCancel }) {
  const [name, setName] = useState(prompt?.name || '')
  const [content, setContent] = useState(prompt?.content || '')

  return (
    <div className="inline-add">
      <div className="inline-add-title">{prompt ? 'Edit Prompt' : 'New Prompt'}</div>
      <Field label="Name" value={name} onChange={setName} placeholder="e.g. Interviewer v2" />
      <div className="form-group">
        <label className="form-label">Content</label>
        <textarea className="form-textarea" style={{ minHeight: 200 }} value={content} onChange={e => setContent(e.target.value)} />
      </div>
      <div className="flex gap-8">
        <button className="btn btn-primary btn-sm" onClick={() => onSave({ name: name.trim(), content })}>Save</button>
        <button className="btn btn-ghost btn-sm" onClick={onCancel}>Cancel</button>
      </div>
    </div>
  )
}

// ── MCP Servers ────────────────────────────────────────────────────────────────

function McpTab() {
  const [servers, setServers] = useState([])
  const [adding, setAdding] = useState(false)

  const load = () => getMcpServers().then(setServers)
  useEffect(() => { load() }, [])

  async function handleCreate(data) { await createMcpServer(data); setAdding(false); load() }
  async function handleDelete(id) { if (!confirm('Delete this MCP server?')) return; await deleteMcpServer(id); load() }

  return (
    <div>
      <div className="flex items-center gap-8 mb-16">
        <h3 style={{ flex: 1 }}>MCP Servers</h3>
        <button className="btn btn-primary btn-sm" onClick={() => setAdding(!adding)}>
          {adding ? 'Cancel' : '+ Add Server'}
        </button>
      </div>

      {adding && (
        <McpForm onSave={handleCreate} onCancel={() => setAdding(false)} />
      )}

      <div className="store-list">
        {servers.map(s => (
          <div key={s.id} className="store-card">
            <div className="store-card-info">
              <div className="store-card-name">{s.name}</div>
              <div className="store-card-meta">{s.endpoint_url}</div>
            </div>
            <div className="store-card-actions">
              <button className="btn btn-danger btn-sm" onClick={() => handleDelete(s.id)}>Del</button>
            </div>
          </div>
        ))}
        {servers.length === 0 && !adding && <div className="text-muted text-small">No MCP servers configured.</div>}
      </div>
    </div>
  )
}

function McpForm({ onSave, onCancel }) {
  const [name, setName] = useState('')
  const [url, setUrl] = useState('')
  return (
    <div className="inline-add">
      <div className="inline-add-title">New MCP Server</div>
      <Field label="Name" value={name} onChange={setName} placeholder="e.g. Memory Server" />
      <Field label="Endpoint URL" value={url} onChange={setUrl} placeholder="http://mcp-server:8080" />
      <div className="flex gap-8">
        <button className="btn btn-primary btn-sm" onClick={() => onSave({ name: name.trim(), endpoint_url: url.trim() })}>Save</button>
        <button className="btn btn-ghost btn-sm" onClick={onCancel}>Cancel</button>
      </div>
    </div>
  )
}

// ── Tests ──────────────────────────────────────────────────────────────────────

function TestsTab() {
  const [experiments, setExperiments] = useState([])
  const [models, setModels] = useState([])
  const [prompts, setPrompts] = useState([])
  const [mcpServers, setMcpServers] = useState([])
  const [globalCfg, setGlobalCfg] = useState(null)
  const [adding, setAdding] = useState(false)
  const [editing, setEditing] = useState(null)

  const load = () => Promise.all([
    getExperiments(), getModels(), getPrompts(), getMcpServers(), getGlobalConfig()
  ]).then(([e, m, p, s, g]) => {
    setExperiments(e); setModels(m); setPrompts(p); setMcpServers(s); setGlobalCfg(g)
  })
  useEffect(() => { load() }, [])

  async function handleCreate(data) { await createExperiment(data); setAdding(false); load() }
  async function handleUpdate(id, data) { await updateExperiment(id, data); setEditing(null); load() }
  async function handleDelete(id) { if (!confirm('Delete this test?')) return; await deleteExperiment(id); load() }

  return (
    <div>
      <div className="flex items-center gap-8 mb-16">
        <h3 style={{ flex: 1 }}>Tests</h3>
        <button className="btn btn-primary btn-sm" onClick={() => setAdding(!adding)}>
          {adding ? 'Cancel' : '+ New Test'}
        </button>
      </div>

      {adding && (
        <TestForm
          models={models} prompts={prompts} mcpServers={mcpServers} globalCfg={globalCfg}
          onSave={handleCreate} onCancel={() => setAdding(false)}
        />
      )}

      <div className="store-list">
        {experiments.map(exp => (
          <div key={exp.id} className="store-card">
            <div className="store-card-info">
              <div className="store-card-name">{exp.name}</div>
              <div className="store-card-meta">
                Pass1: {modelName(models, exp.pass1_model_id)} ·
                Pass2: {modelName(models, exp.pass2_model_id)} ·
                IV: {modelName(models, exp.interviewer_model_id)}
              </div>
            </div>
            <div className="store-card-actions">
              <button className="btn btn-ghost btn-sm" onClick={() => setEditing(exp.id)}>Edit</button>
              <button className="btn btn-danger btn-sm" onClick={() => handleDelete(exp.id)}>Del</button>
            </div>
          </div>
        ))}
        {experiments.length === 0 && !adding && (
          <div className="text-muted text-small">No tests configured. Add models first, then create a test.</div>
        )}
      </div>

      {editing && (
        <div className="overlay" onClick={() => setEditing(null)}>
          <div className="config-panel" style={{ width: 'min(700px,90vw)' }} onClick={e => e.stopPropagation()}>
            <div className="config-panel-header">
              Edit Test
              <button className="btn btn-ghost btn-sm" onClick={() => setEditing(null)}>✕</button>
            </div>
            <div className="config-panel-body">
              <TestForm
                experiment={experiments.find(e => e.id === editing)}
                models={models} prompts={prompts} mcpServers={mcpServers} globalCfg={globalCfg}
                onSave={d => handleUpdate(editing, d)}
                onCancel={() => setEditing(null)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function TestForm({ experiment, models: initialModels, prompts: initialPrompts, mcpServers, globalCfg, onSave, onCancel }) {
  const [name, setName] = useState(experiment?.name || '')
  const [pass1, setPass1] = useState(experiment?.pass1_model_id || '')
  const [pass2, setPass2] = useState(experiment?.pass2_model_id || '')
  const [iv, setIv] = useState(experiment?.interviewer_model_id || '')
  const [turnLimit, setTurnLimit] = useState(experiment?.turn_limit || '')
  const [ctxWindow, setCtxWindow] = useState(experiment?.context_window || '')
  const [compaction, setCompaction] = useState(experiment?.compaction_threshold_fraction || '')
  const [pass1Act, setPass1Act] = useState(experiment?.pass1_activation_fraction || '')
  const [turnPauseMin, setTurnPauseMin] = useState(experiment?.turn_pause_min_seconds || '')
  const [turnPauseMax, setTurnPauseMax] = useState(experiment?.turn_pause_max_seconds || '')
  const [selectedMcp, setSelectedMcp] = useState(experiment?.mcp_server_ids || [])

  // Local stores — start from props, can be augmented inline
  const [localModels, setLocalModels] = useState(initialModels)
  const [localPrompts, setLocalPrompts] = useState(initialPrompts)
  const [addingModelFor, setAddingModelFor] = useState(null) // 'pass1' | 'pass2' | 'iv'
  const [addingPrompt, setAddingPrompt] = useState(false)

  async function handleAddModel(data) {
    const m = await createModel(data)
    setLocalModels(prev => [...prev, m])
    if (addingModelFor === 'pass1') setPass1(m.id)
    if (addingModelFor === 'pass2') setPass2(m.id)
    if (addingModelFor === 'iv') setIv(m.id)
    setAddingModelFor(null)
  }

  async function handleAddPrompt(data) {
    const p = await createPrompt(data)
    setLocalPrompts(prev => [...prev, p])
    setAddingPrompt(false)
  }

  function toggleMcp(id) {
    setSelectedMcp(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])
  }

  function save() {
    if (!name.trim() || !pass1 || !pass2 || !iv) {
      alert('Name and all three model selections are required.')
      return
    }
    onSave({
      name: name.trim(),
      pass1_model_id: pass1,
      pass2_model_id: pass2,
      interviewer_model_id: iv,
      mcp_server_ids: selectedMcp,
      turn_limit: turnLimit ? +turnLimit : null,
      context_window: ctxWindow ? +ctxWindow : null,
      compaction_threshold_fraction: compaction ? +compaction : null,
      pass1_activation_fraction: pass1Act ? +pass1Act : null,
      turn_pause_min_seconds: turnPauseMin !== '' ? +turnPauseMin : null,
      turn_pause_max_seconds: turnPauseMax !== '' ? +turnPauseMax : null,
    })
  }

  return (
    <div>
      <Field label="Test Name" value={name} onChange={setName} placeholder="e.g. Qwen3.5 Context Test" />
      <div className="divider" />
      <h4 style={{ marginBottom: 12, color: 'var(--text2)', fontSize: 12, textTransform: 'uppercase' }}>Models</h4>
      <p className="text-muted text-small" style={{ marginBottom: 12 }}>
        All model selections are logged in output files for full audit traceability.
      </p>
      <div className="form-row">
        <ModelSelect label="Pass 1 Model" value={pass1} onChange={setPass1} models={localModels}
          onAddNew={() => setAddingModelFor(addingModelFor === 'pass1' ? null : 'pass1')} />
        <ModelSelect label="Pass 2 Model (Test Model)" value={pass2} onChange={setPass2} models={localModels}
          onAddNew={() => setAddingModelFor(addingModelFor === 'pass2' ? null : 'pass2')} />
      </div>
      {addingModelFor === 'pass1' || addingModelFor === 'pass2' ? (
        <ModelForm
          onSave={handleAddModel}
          onCancel={() => setAddingModelFor(null)}
        />
      ) : null}
      <ModelSelect label="Interviewer Model" value={iv} onChange={setIv} models={localModels}
        onAddNew={() => setAddingModelFor(addingModelFor === 'iv' ? null : 'iv')} />
      {addingModelFor === 'iv' && (
        <ModelForm onSave={handleAddModel} onCancel={() => setAddingModelFor(null)} />
      )}

      <div className="divider" />
      <h4 style={{ marginBottom: 12, color: 'var(--text2)', fontSize: 12, textTransform: 'uppercase' }}>Parameter Overrides</h4>
      <p className="text-muted text-small" style={{ marginBottom: 12 }}>Leave blank to use global defaults.</p>
      <div className="form-row">
        <Field label="Turn Limit" value={turnLimit} onChange={setTurnLimit} type="number" placeholder={globalCfg ? String(globalCfg.turn_limit) : 'default'} />
        <Field label="Context Window" value={ctxWindow} onChange={setCtxWindow} type="number" placeholder={globalCfg ? String(globalCfg.context_window) : 'default'} />
      </div>
      <div className="form-row">
        <Field label="Compaction Threshold" value={compaction} onChange={setCompaction} type="number" step="0.01" placeholder={globalCfg ? String(globalCfg.compaction_threshold_fraction) : 'default'} />
        <Field label="Pass 1 Activation" value={pass1Act} onChange={setPass1Act} type="number" step="0.01" placeholder={globalCfg ? String(globalCfg.pass1_activation_fraction) : 'default'} />
      </div>
      <div className="form-row">
        <Field label="Turn Pause Min (seconds)" value={turnPauseMin} onChange={setTurnPauseMin} type="number" step="1" placeholder={globalCfg ? String(globalCfg.turn_pause_min_seconds) : 'default'} />
        <Field label="Turn Pause Max (seconds)" value={turnPauseMax} onChange={setTurnPauseMax} type="number" step="1" placeholder={globalCfg ? String(globalCfg.turn_pause_max_seconds) : 'default'} />
      </div>

      {mcpServers.length > 0 && (
        <>
          <div className="divider" />
          <h4 style={{ marginBottom: 12, color: 'var(--text2)', fontSize: 12, textTransform: 'uppercase' }}>MCP Servers</h4>
          {mcpServers.map(s => (
            <label key={s.id} style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8, cursor: 'pointer' }}>
              <input type="checkbox" checked={selectedMcp.includes(s.id)} onChange={() => toggleMcp(s.id)} />
              <span>{s.name}</span>
              <span className="text-muted text-small">{s.endpoint_url}</span>
            </label>
          ))}
        </>
      )}

      <div className="divider" />
      <div className="flex items-center gap-8 mb-16">
        <h4 style={{ flex: 1, color: 'var(--text2)', fontSize: 12, textTransform: 'uppercase', margin: 0 }}>Prompt Store</h4>
        <button className="btn btn-ghost btn-sm" onClick={() => setAddingPrompt(!addingPrompt)}>
          {addingPrompt ? 'Cancel' : '+ Add Prompt'}
        </button>
      </div>
      <p className="text-muted text-small" style={{ marginBottom: 12 }}>
        Store prompts override built-in defaults for all tests.
      </p>
      {localPrompts.length > 0 && (
        <div className="store-list" style={{ marginBottom: 12 }}>
          {localPrompts.map(p => (
            <div key={p.id} className="store-card" style={{ padding: '8px 12px' }}>
              <div className="store-card-name" style={{ fontSize: 13 }}>{p.name}</div>
              <div className="store-card-meta">{p.content.slice(0, 60)}{p.content.length > 60 ? '…' : ''}</div>
            </div>
          ))}
        </div>
      )}
      {localPrompts.length === 0 && !addingPrompt && (
        <p className="text-muted text-small" style={{ marginBottom: 12 }}>No overrides — built-in defaults are used.</p>
      )}
      {addingPrompt && (
        <PromptForm onSave={handleAddPrompt} onCancel={() => setAddingPrompt(false)} />
      )}

      <div className="divider" />
      <div className="flex gap-8">
        <button className="btn btn-primary" onClick={save}>Save Test</button>
        <button className="btn btn-ghost" onClick={onCancel}>Cancel</button>
      </div>
    </div>
  )
}

// ── Shared components ──────────────────────────────────────────────────────────

function Field({ label, value, onChange, type = 'text', placeholder, step }) {
  return (
    <div className="form-group">
      <label className="form-label">{label}</label>
      <input
        className="form-input"
        type={type}
        step={step}
        value={value}
        placeholder={placeholder}
        onChange={e => onChange(e.target.value)}
      />
    </div>
  )
}

function ModelSelect({ label, value, onChange, models, onAddNew }) {
  return (
    <div className="form-group">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <label className="form-label" style={{ margin: 0 }}>{label}</label>
        {onAddNew && (
          <button
            className="btn btn-ghost btn-sm"
            style={{ fontSize: 11, padding: '1px 6px', lineHeight: 1.4 }}
            onClick={onAddNew}
          >
            + New
          </button>
        )}
      </div>
      <select className="form-select" value={value} onChange={e => onChange(e.target.value)}>
        <option value="">— Select model —</option>
        {models.map(m => (
          <option key={m.id} value={m.id}>{m.name} ({m.model_identifier})</option>
        ))}
      </select>
    </div>
  )
}

function KeyDialog({ label, onSave, onClose }) {
  const [key, setKey] = useState('')
  const [show, setShow] = useState(false)
  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog" onClick={e => e.stopPropagation()}>
        <div className="dialog-title">{label}</div>
        <div className="dialog-body">
          <div className="flex gap-8">
            <input
              className="form-input flex-1"
              type={show ? 'text' : 'password'}
              placeholder="Enter API key…"
              value={key}
              onChange={e => setKey(e.target.value)}
              autoFocus
            />
            <button className="btn btn-ghost btn-sm" onClick={() => setShow(!show)}>
              {show ? 'Hide' : 'Show'}
            </button>
          </div>
        </div>
        <div className="dialog-actions">
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={() => onSave(key)}>Save Key</button>
        </div>
      </div>
    </div>
  )
}

function modelName(models, id) {
  return models.find(m => m.id === id)?.name || id?.slice(0, 8) || '—'
}
