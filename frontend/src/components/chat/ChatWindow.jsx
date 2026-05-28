import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble.jsx'
import TypingIndicator from './TypingIndicator.jsx'

export default function ChatWindow({ messages, isWaiting }) {
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isWaiting])

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="max-w-3xl mx-auto">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        {isWaiting && <TypingIndicator />}
        <div ref={endRef} />
      </div>
    </div>
  )
}
