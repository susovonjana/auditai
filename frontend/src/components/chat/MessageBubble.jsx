import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useState } from 'react'
import { sendFeedback } from '../../api.js'

export default function MessageBubble({ message }) {
  const [feedback, setFeedback] = useState(message.feedback || null)
  const [submitting, setSubmitting] = useState(false)
  const isUser = message.role === 'user'

  const submit = async (value) => {
    if (!message.historyId || submitting) return
    setSubmitting(true)
    try {
      await sendFeedback(message.historyId, value)
      setFeedback(value)
    } catch (_) {
      // silent — feedback is best-effort
    } finally {
      setSubmitting(false)
    }
  }

  if (isUser) {
    return (
      <div className="flex justify-end my-3">
        <div className="bg-brand-600 text-white px-4 py-3 rounded-2xl rounded-tr-sm max-w-[80%] shadow-sm">
          <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start my-3">
      <div className="bg-white border border-gray-200 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[85%] shadow-sm">
        <div className="markdown text-gray-800">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>

        {message.documentsReferenced?.length > 0 && (
          <div className="mt-3 text-xs text-gray-500 border-t border-gray-100 pt-2">
            <span className="font-medium">Sources: </span>
            {message.documentsReferenced.join(', ')}
          </div>
        )}

        {message.historyId && (
          <div className="mt-3 flex items-center gap-2 text-xs">
            <button
              type="button"
              onClick={() => submit('helpful')}
              disabled={submitting || feedback === 'helpful'}
              className={`px-2 py-1 rounded border transition ${
                feedback === 'helpful'
                  ? 'bg-green-50 border-green-400 text-green-700'
                  : 'border-gray-200 text-gray-500 hover:bg-gray-50'
              }`}
              aria-label="Helpful"
            >
              👍 Helpful
            </button>
            <button
              type="button"
              onClick={() => submit('not_helpful')}
              disabled={submitting || feedback === 'not_helpful'}
              className={`px-2 py-1 rounded border transition ${
                feedback === 'not_helpful'
                  ? 'bg-red-50 border-red-400 text-red-700'
                  : 'border-gray-200 text-gray-500 hover:bg-gray-50'
              }`}
              aria-label="Not helpful"
            >
              👎 Not Helpful
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
