// API client — all calls to the FastAPI backend

const BASE = '/api'

async function req(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body !== undefined) opts.body = JSON.stringify(body)
  const res = await fetch(BASE + path, opts)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${method} ${path} → ${res.status}: ${text}`)
  }
  if (res.status === 204) return null
  return res.json()
}

// ── Models ─────────────────────────────────────────────────────────────────
export const getModels = () => req('GET', '/models')
export const createModel = (data) => req('POST', '/models', data)
export const updateModel = (id, data) => req('PATCH', `/models/${id}`, data)
export const deleteModel = (id) => req('DELETE', `/models/${id}`)
export const setModelKey = (id, key) => req('PUT', `/models/${id}/key`, { key })

// ── Prompts ────────────────────────────────────────────────────────────────
export const getDefaultPrompts = () => req('GET', '/prompts/defaults')
export const getPrompts = () => req('GET', '/prompts')
export const createPrompt = (data) => req('POST', '/prompts', data)
export const updatePrompt = (id, data) => req('PATCH', `/prompts/${id}`, data)
export const deletePrompt = (id) => req('DELETE', `/prompts/${id}`)

// ── MCP Servers ────────────────────────────────────────────────────────────
export const getMcpServers = () => req('GET', '/mcp-servers')
export const createMcpServer = (data) => req('POST', '/mcp-servers', data)
export const updateMcpServer = (id, data) => req('PATCH', `/mcp-servers/${id}`, data)
export const deleteMcpServer = (id) => req('DELETE', `/mcp-servers/${id}`)

// ── Experiments ────────────────────────────────────────────────────────────
export const getExperiments = () => req('GET', '/experiments')
export const createExperiment = (data) => req('POST', '/experiments', data)
export const updateExperiment = (id, data) => req('PATCH', `/experiments/${id}`, data)
export const deleteExperiment = (id) => req('DELETE', `/experiments/${id}`)

// ── Global config ──────────────────────────────────────────────────────────
export const getGlobalConfig = () => req('GET', '/config/globals')
export const updateGlobalConfig = (data) => req('PATCH', '/config/globals', data)
export const getSearchConfig = () => req('GET', '/config/search')
export const updateSearchConfig = (data) => req('PUT', '/config/search', data)

// ── Runs ───────────────────────────────────────────────────────────────────
export const getRuns = () => req('GET', '/runs')
export const startRun = (data) => req('POST', '/runs', data)
export const cancelRun = (id) => req('DELETE', `/runs/${id}`)
export const finishRun = (id) => req('PATCH', `/runs/${id}`)
export const getTranscript = (id) => fetch(`${BASE}/runs/${id}/transcript`).then(r => r.text())
export const getEvents = (id) => req('GET', `/runs/${id}/events`)

// SSE stream — returns an EventSource
export function streamRun(runId, onEvent, onDone, onError) {
  const es = new EventSource(`${BASE}/runs/${runId}/stream`)
  es.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data)
      onEvent(event)
      if (event.type === 'run_complete' || event.type === 'error' || event.type === 'cancelled') {
        es.close()
        onDone(event)
      }
    } catch (err) {
      console.error('SSE parse error', err)
    }
  }
  es.onerror = (err) => {
    es.close()
    onError(err)
  }
  return es
}
