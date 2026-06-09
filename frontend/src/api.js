import axios from 'axios'

const api = axios.create({
  baseURL: '',           // dev proxy handles routing; in prod set VITE_API_BASE_URL
  timeout: 120000,
})

// Attach admin JWT to every /admin request automatically
api.interceptors.request.use((config) => {
  if (config.url && config.url.startsWith('/admin')) {
    const token = localStorage.getItem('auditai_admin_token')
    if (token) config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// On 401 from an admin endpoint, drop the token and bounce to login
api.interceptors.response.use(
  (resp) => resp,
  (err) => {
    if (
      err?.response?.status === 401 &&
      err?.config?.url?.startsWith('/admin')
    ) {
      localStorage.removeItem('auditai_admin_token')
      if (!window.location.pathname.startsWith('/admin/login')) {
        window.location.href = '/admin/login?expired=1'
      }
    }
    return Promise.reject(err)
  },
)

// ============================== User chat ==============================
export const startSession = () => api.post('/session/start')
export const askQuestion = (sessionToken, question) =>
  api.post('/ask', { session_token: sessionToken, question })
export const sendFeedback = (historyId, feedback) =>
  api.post('/feedback', { history_id: historyId, feedback })
export const getSessionHistory = (token) =>
  api.get(`/session/${token}/history`)

/**
 * Stream a /ask/stream response. Calls callbacks as events arrive:
 *   onMeta({documents, chunks_found})
 *   onDelta(textPiece)
 *   onDone({history_id, was_answered, response_time_ms, documents})
 *   onError(message)  — also fires for 429 rate limit + 413 too long
 *
 * Returns an AbortController so callers can cancel mid-stream.
 *
 * If no chunk arrives for `stallTimeoutMs` (default 60s) the stream is
 * aborted and onError fires with a stall message. This prevents a hung
 * server connection from sitting forever.
 */
export async function askQuestionStream(
  sessionToken,
  question,
  { onMeta, onDelta, onDone, onError, language = 'en', stallTimeoutMs = 60000 } = {},
) {
  const controller = new AbortController()
  let stalled = false
  let stallTimer = null
  const resetStallTimer = () => {
    if (stallTimer) clearTimeout(stallTimer)
    stallTimer = setTimeout(() => {
      stalled = true
      try { controller.abort() } catch {}
      onError?.('The response stalled — no data for 60s. Please try again.')
    }, stallTimeoutMs)
  }
  const clearStallTimer = () => {
    if (stallTimer) { clearTimeout(stallTimer); stallTimer = null }
  }

  ;(async () => {
    let resp
    resetStallTimer()
    try {
      resp = await fetch('/ask/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_token: sessionToken,
          question,
          language,
        }),
        signal: controller.signal,
      })
    } catch (e) {
      clearStallTimer()
      if (!stalled) onError?.('Network error contacting AuditAI.')
      return
    }
    if (!resp.ok || !resp.body) {
      clearStallTimer()
      // Try to surface the server-side reason cleanly
      let detail = `Server error (${resp.status}).`
      try {
        const j = await resp.json()
        if (j?.detail) detail = j.detail
        else if (j?.error) detail = j.error
      } catch {}
      if (resp.status === 429) {
        detail =
          detail ||
          'Too many requests. Please slow down and try again in a moment.'
      } else if (resp.status === 413) {
        detail = detail || 'Your question is too long. Please shorten it.'
      } else if (resp.status === 503 || resp.status === 502) {
        detail =
          detail ||
          'AuditAI is having trouble reaching the AI service. Please try again.'
      }
      onError?.(detail)
      return
    }
    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    try {
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        resetStallTimer()
        buffer += decoder.decode(value, { stream: true })
        let nl
        while ((nl = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, nl).trim()
          buffer = buffer.slice(nl + 1)
          if (!line) continue
          let event
          try {
            event = JSON.parse(line)
          } catch {
            continue
          }
          if (event.type === 'meta') onMeta?.(event)
          else if (event.type === 'delta') onDelta?.(event.text || '')
          else if (event.type === 'done') onDone?.(event)
          else if (event.type === 'error') onError?.(event.message || 'Stream error.')
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError' && !stalled) onError?.('Stream interrupted.')
    } finally {
      clearStallTimer()
    }
  })()
  return controller
}

// ============================== Admin ==============================
export const adminLogin = (username, password) =>
  api.post('/admin/login', { username, password })
export const adminLogout = () => api.post('/admin/logout')
export const getKbStatus = () => api.get('/admin/status')

export const uploadDocument = (file, category, onProgress) => {
  const form = new FormData()
  form.append('file', file)
  if (category) form.append('category', category)
  return api.post('/admin/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded * 100) / e.total))
      }
    },
  })
}

export const listDocuments = () => api.get('/admin/documents')
export const getDocument = (id) => api.get(`/admin/documents/${id}`)
export const updateDocument = (id, payload) =>
  api.patch(`/admin/documents/${id}`, payload)
export const deleteDocument = (id) => api.delete(`/admin/documents/${id}`)

export const getSearchHistory = (params = {}) =>
  api.get('/admin/search-history', { params })
export const getUnanswered = (params = {}) =>
  api.get('/admin/search-history/unanswered', { params })
export const exportSearchHistory = () =>
  api.get('/admin/search-history/export', { responseType: 'blob' })

export const getAnalyticsSummary = () => api.get('/admin/analytics/summary')
export const getTopTopics = (limit = 10) =>
  api.get('/admin/analytics/top-topics', { params: { limit } })

export const listSessions = () => api.get('/admin/sessions')
export const getSessionDetail = (id) => api.get(`/admin/sessions/${id}`)

export default api
