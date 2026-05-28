import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { askQuestion, startSession } from '../api.js'
import ChatWindow from '../components/chat/ChatWindow.jsx'
import QuestionInput from '../components/chat/QuestionInput.jsx'
import SuggestedQuestions from '../components/chat/SuggestedQuestions.jsx'

const SESSION_KEY = 'auditai_session_token'

export default function ChatPage() {
  const [sessionToken, setSessionToken] = useState(null)
  const [messages, setMessages] = useState([])
  const [isWaiting, setIsWaiting] = useState(false)

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
      .catch(() => {
        // best-effort — UI will still render
      })
  }, [])

  const handleSend = async (question) => {
    if (!sessionToken) return
    const userMsg = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: question,
    }
    setMessages((prev) => [...prev, userMsg])
    setIsWaiting(true)

    try {
      const resp = await askQuestion(sessionToken, question)
      const ai = resp.data
      setMessages((prev) => [
        ...prev,
        {
          id: `a-${ai.history_id}`,
          role: 'assistant',
          content: ai.answer,
          historyId: ai.history_id,
          documentsReferenced: ai.documents_referenced,
        },
      ])
    } catch (_) {
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: 'assistant',
          content:
            'There was a problem generating a response. Please try again.',
        },
      ])
    } finally {
      setIsWaiting(false)
    }
  }

  const hasMessages = messages.length > 0

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Top bar */}
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

      {/* Body */}
      {hasMessages ? (
        <ChatWindow messages={messages} isWaiting={isWaiting} />
      ) : (
        <div className="flex-1 flex items-center justify-center py-8">
          <SuggestedQuestions onSelect={handleSend} />
        </div>
      )}

      {/* Input */}
      <QuestionInput onSend={handleSend} disabled={isWaiting || !sessionToken} />
    </div>
  )
}
