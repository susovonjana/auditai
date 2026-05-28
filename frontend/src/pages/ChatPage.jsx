import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { askQuestionStream, startSession } from '../api.js'
import ChatWindow from '../components/chat/ChatWindow.jsx'
import QuestionInput from '../components/chat/QuestionInput.jsx'
import SuggestedQuestions from '../components/chat/SuggestedQuestions.jsx'

const SESSION_KEY = 'auditai_session_token'

export default function ChatPage() {
  const [sessionToken, setSessionToken] = useState(null)
  const [messages, setMessages] = useState([])
  const [isWaiting, setIsWaiting] = useState(false)
  const streamRef = useRef(null)

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

  const updateMessage = (id, patch) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    )
  }

  const appendDelta = (id, text) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, content: (m.content || '') + text } : m,
      ),
    )
  }

  const handleSend = async (question) => {
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
    }
    setMessages((prev) => [...prev, userMsg, aiMsg])
    setIsWaiting(true)

    streamRef.current = await askQuestionStream(sessionToken, question, {
      onMeta: (m) => {
        updateMessage(aiId, { documentsReferenced: m.documents || [] })
      },
      onDelta: (t) => appendDelta(aiId, t),
      onDone: (d) => {
        updateMessage(aiId, {
          streaming: false,
          historyId: d.history_id,
          documentsReferenced: d.documents || [],
          wasAnswered: d.was_answered,
        })
        setIsWaiting(false)
      },
      onError: (err) => {
        updateMessage(aiId, {
          streaming: false,
          content:
            (aiMsg.content || '') +
            `\n\n_There was a problem generating a response. Please try again. (${err})_`,
        })
        setIsWaiting(false)
      },
    })
  }

  const hasMessages = messages.length > 0

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-4 py-3">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-lg">⚖️</span>
            <span className="font-semibold text-gray-900">AuditAI Assistant</span>
          </div>
          <Link
            to="/admin/login"
            className="text-xs text-gray-400 hover:text-brand-700"
          >
            Admin
          </Link>
        </div>
      </header>

      {hasMessages ? (
        <ChatWindow messages={messages} isWaiting={isWaiting} />
      ) : (
        <div className="flex-1 flex items-center justify-center py-8">
          <SuggestedQuestions onSelect={handleSend} />
        </div>
      )}

      <QuestionInput onSend={handleSend} disabled={isWaiting || !sessionToken} />
    </div>
  )
}
