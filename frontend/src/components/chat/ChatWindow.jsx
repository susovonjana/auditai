import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble.jsx'
import TypingIndicator from './TypingIndicator.jsx'

export default function ChatWindow({ messages, isWaiting, onSendQuestion }) {
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isWaiting])

  // A streaming AI message with no content yet is shown as a typing indicator;
  // once any text arrives, it switches to a regular bubble.
  const last = messages[messages.length - 1]
  const showTyping =
    isWaiting && last?.role === 'assistant' && !(last.content || '').trim()

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="max-w-3xl mx-auto">
        {messages.map((m) => {
          if (m.role === 'assistant' && !(m.content || '').trim()) return null
          return (
            <MessageBubble
              key={m.id}
              message={m}
              onSendQuestion={onSendQuestion}
              isBusy={isWaiting}
            />
          )
        })}
        {showTyping && <TypingIndicator />}
        <div ref={endRef} />
      </div>
    </div>
  )
}
