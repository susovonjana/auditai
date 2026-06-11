import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { askQuestionStream, startSession } from '../api.js'
import ChatWindow from '../components/chat/ChatWindow.jsx'
import LanguageToggle from '../components/chat/LanguageToggle.jsx'
import QuestionInput from '../components/chat/QuestionInput.jsx'
import SuggestedQuestions from '../components/chat/SuggestedQuestions.jsx'
import { getLanguage, t } from '../i18n.js'

const SESSION_KEY = 'auditai_session_token'

// How often to flush buffered streaming tokens into React state.
// Lower = smoother UI; higher = fewer re-renders. 50ms is fast enough to feel
// real-time but coarse enough to keep React happy on 5k-token answers.
const STREAM_FLUSH_MS = 50

export default function ChatPage() {
  const [sessionToken, setSessionToken] = useState(null)
  const [messages, setMessages] = useState([])
  const [isWaiting, setIsWaiting] = useState(false)
  const [lang, setLang] = useState(getLanguage())
  const streamRef = useRef(null)
  // Pending text per in-flight assistant message — flushed to React state
  // every STREAM_FLUSH_MS so we don't trigger an O(n²) re-render storm.
  const pendingRef = useRef({ id: null, buf: '', timer: null })

  useEffect(() => {
    const handler = (e) => setLang(e.detail)
    window.addEventListener('auditai-lang-change', handler)
    return () => window.removeEventListener('auditai-lang-change', handler)
  }, [])

  // Acquire (or reuse) a session token on mount
  useEffect(() => {
    const stored = localStorage.getItem(SESSION_KEY)
    if (stored) {
      setSessionToken(stored)
      return
    }
    startSession()
      .then((r) => {
        localStorage.setItem(SESSION_KEY, r.data.session_token)
        setSessionToken(r.data.session_token)
      })
      .catch(() => {})
  }, [])

  // Abort any in-flight stream when the page unmounts.
  useEffect(() => {
    return () => {
      try { streamRef.current?.abort?.() } catch {}
      if (pendingRef.current.timer) clearTimeout(pendingRef.current.timer)
    }
  }, [])

  const updateMessage = useCallback((id, patch) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    )
  }, [])

  const flushPending = useCallback(() => {
    const p = pendingRef.current
    if (!p.id || !p.buf) return
    const chunk = p.buf
    const id = p.id
    p.buf = ''
    p.timer = null
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, content: (m.content || '') + chunk } : m,
      ),
    )
  }, [])

  const queueDelta = useCallback((id, text) => {
    const p = pendingRef.current
    if (p.id !== id) {
      // New stream; flush any leftovers and switch target
      flushPending()
      p.id = id
    }
    p.buf += text
    if (!p.timer) {
      p.timer = setTimeout(flushPending, STREAM_FLUSH_MS)
    }
  }, [flushPending])

  const finalizeStream = useCallback(() => {
    if (pendingRef.current.timer) {
      clearTimeout(pendingRef.current.timer)
      pendingRef.current.timer = null
    }
    flushPending()
  }, [flushPending])

  const handleStop = useCallback(() => {
    try { streamRef.current?.abort?.() } catch {}
    finalizeStream()
    setMessages((prev) =>
      prev.map((m) => (m.streaming ? { ...m, streaming: false, stopped: true } : m)),
    )
    setIsWaiting(false)
  }, [finalizeStream])

  // Recover from a stale session token: clear it, ask the backend for a new
  // one, persist, and return it. Used when /ask answers 404 "Session not found".
  const refreshSession = useCallback(async () => {
    localStorage.removeItem(SESSION_KEY)
    const r = await startSession()
    const fresh = r.data.session_token
    localStorage.setItem(SESSION_KEY, fresh)
    setSessionToken(fresh)
    return fresh
  }, [])

  const _runStream = useCallback((tokenToUse, question, aiId, attempt) =>
    askQuestionStream(tokenToUse, question, {
      language: lang,
      onMeta: (m) => {
        updateMessage(aiId, { documentsReferenced: m.documents || [] })
      },
      onDelta: (text) => queueDelta(aiId, text),
      onDone: (d) => {
        finalizeStream()
        updateMessage(aiId, {
          streaming: false,
          historyId: d.history_id,
          documentsReferenced: d.documents || [],
          wasAnswered: d.was_answered,
        })
        setIsWaiting(false)
      },
      onError: async (err) => {
        // Auto-recover on stale session: refresh and retry once.
        if (attempt === 0 && typeof err === 'string' && err.toLowerCase().includes('session')) {
          try {
            const fresh = await refreshSession()
            streamRef.current = await _runStream(fresh, question, aiId, attempt + 1)
            return
          } catch {
            // fall through to surfacing the original error
          }
        }
        finalizeStream()
        updateMessage(aiId, {
          streaming: false,
          error: err || t('error_generic', lang),
        })
        setIsWaiting(false)
      },
    }),
    [lang, updateMessage, queueDelta, finalizeStream, refreshSession],
  )

  const handleSend = useCallback(async (question) => {
    if (!sessionToken || isWaiting) return

    const userMsg = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: question,
    }
    const aiId = `a-${Date.now()}`
    const aiMsg = {
      id: aiId,
      role: 'assistant',
      content: '',
      streaming: true,
      documentsReferenced: [],
      sourceQuestion: question,  // remembered so Retry can re-fire it
    }
    setMessages((prev) => [...prev, userMsg, aiMsg])
    setIsWaiting(true)

    streamRef.current = await _runStream(sessionToken, question, aiId, 0)
  }, [sessionToken, isWaiting, _runStream])

  // Re-run the same question that errored. Drops the failed assistant
  // message so the chat history stays tidy.
  const handleRetry = useCallback((failedMessageId, question) => {
    setMessages((prev) => prev.filter((m) => m.id !== failedMessageId))
    // Also drop the user message paired with it so handleSend re-adds a fresh pair
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.role === 'user' && m.content === question)
      if (idx === -1) return prev
      // Only strip if it's the most recent matching user msg (right before the failed AI msg)
      return prev.slice(0, idx).concat(prev.slice(idx + 1))
    })
    handleSend(question)
  }, [handleSend])

  const hasMessages = messages.length > 0

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-4 py-3">
        <div className="max-w-3xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="text-lg">⚖️</span>
            <span className="font-semibold text-gray-900">{t('app_title', lang)}</span>
          </div>
          <div className="flex items-center gap-3">
            <LanguageToggle />
            <Link
              to="/admin/login"
              className="text-xs text-gray-400 hover:text-brand-700"
            >
              {t('admin_link', lang)}
            </Link>
          </div>
        </div>
      </header>

      {hasMessages ? (
        <ChatWindow
          messages={messages}
          isWaiting={isWaiting}
          onSendQuestion={handleSend}
          onRetry={handleRetry}
        />
      ) : (
        <div className="flex-1 flex items-center justify-center py-8">
          <SuggestedQuestions onSelect={handleSend} />
        </div>
      )}

      <QuestionInput
        onSend={handleSend}
        onStop={handleStop}
        isWaiting={isWaiting}
        disabled={!sessionToken}
      />
    </div>
  )
}
