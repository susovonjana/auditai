import { useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { sendFeedback } from '../../api.js'

// Split the AI response into its labelled sections so we can style each
// one distinctly. Recognises the current schema (KB + Follow-ups) AND
// legacy schemas (with Additional context / Key Takeaway) so older
// messages still render correctly.
function splitSections(markdown) {
  if (!markdown) return null
  const re = /^##\s+(From your knowledge base|Additional context|Key Takeaway|Follow[- ]?up Questions?|Follow[- ]?ups?)\s*$/gim
  const matches = [...markdown.matchAll(re)]
  if (matches.length === 0) return null

  const out = { kb: '', extra: '', key: '', followups: [] }
  for (let i = 0; i < matches.length; i++) {
    const m = matches[i]
    const name = m[1].toLowerCase()
    const start = m.index + m[0].length
    const end = i + 1 < matches.length ? matches[i + 1].index : markdown.length
    const body = markdown.slice(start, end).trim()

    if (name.startsWith('from your')) out.kb = body
    else if (name.startsWith('additional')) out.extra = body
    else if (name.startsWith('key')) out.key = body
    else if (name.startsWith('follow')) {
      // Extract each bullet/numbered item as a clean question string
      for (const raw of body.split('\n')) {
        const item = raw.match(/^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$/)
        if (item) {
          const cleaned = item[1]
            .replace(/^\*\*|\*\*$/g, '')      // strip surrounding **bold**
            .replace(/^["'`]+|["'`]+$/g, '')  // strip surrounding quotes
            .trim()
          if (cleaned) out.followups.push(cleaned)
        }
      }
    }
  }
  return out
}

function Section({ accent, title, icon, children, dim }) {
  return (
    <section
      className={`rounded-xl border ${accent} px-4 py-3 mb-3 ${dim ? 'opacity-95' : ''}`}
    >
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide mb-2">
        <span>{icon}</span>
        <span>{title}</span>
      </div>
      <div className="markdown text-gray-800">{children}</div>
    </section>
  )
}

function FollowUps({ questions, onSelect, disabled }) {
  if (!questions || questions.length === 0) return null
  return (
    <div className="mt-3 rounded-xl border bg-indigo-50 border-indigo-100 px-4 py-3">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide mb-2 text-indigo-700">
        <span>💬</span>
        <span>Suggested follow-up questions</span>
      </div>
      <div className="flex flex-col gap-2">
        {questions.map((q, i) => (
          <button
            key={i}
            type="button"
            onClick={() => onSelect?.(q)}
            disabled={disabled || !onSelect}
            className="group text-left px-3 py-2 bg-white hover:bg-indigo-100 border border-indigo-100 hover:border-indigo-300 rounded-lg text-sm text-indigo-800 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-between gap-3"
          >
            <span>{q}</span>
            <span className="text-indigo-400 group-hover:text-indigo-600 text-xs shrink-0">→</span>
          </button>
        ))}
      </div>
    </div>
  )
}

export default function MessageBubble({ message, onSendQuestion, isBusy }) {
  const [feedback, setFeedback] = useState(message.feedback || null)
  const [submitting, setSubmitting] = useState(false)
  const isUser = message.role === 'user'
  const sections = useMemo(() => splitSections(message.content), [message.content])

  const submit = async (value) => {
    if (!message.historyId || submitting) return
    setSubmitting(true)
    try {
      await sendFeedback(message.historyId, value)
      setFeedback(value)
    } catch (_) {
      // best-effort
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
      <div className="bg-white border border-gray-200 px-5 py-4 rounded-2xl rounded-tl-sm max-w-[88%] shadow-sm w-full">
        {sections ? (
          <>
            {sections.kb && (
              <Section
                accent="bg-brand-50 border-brand-100"
                title="From your knowledge base"
                icon="📚"
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {sections.kb}
                </ReactMarkdown>
              </Section>
            )}
            {sections.extra && (
              <Section
                accent="bg-amber-50 border-amber-100"
                title="Additional context"
                icon="💡"
                dim
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {sections.extra}
                </ReactMarkdown>
              </Section>
            )}
            {/* Legacy: render Key Takeaway only if present in older saved messages */}
            {sections.key && (
              <Section
                accent="bg-green-50 border-green-100"
                title="Key takeaway"
                icon="🎯"
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {sections.key}
                </ReactMarkdown>
              </Section>
            )}
            <FollowUps
              questions={sections.followups}
              onSelect={onSendQuestion}
              disabled={isBusy}
            />
          </>
        ) : (
          <div className="markdown text-gray-800">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content || ''}
            </ReactMarkdown>
          </div>
        )}

        {/* {message.documentsReferenced?.length > 0 && (
          <div className="mt-1 text-xs text-gray-500 border-t border-gray-100 pt-2">
            <span className="font-medium">Sources: </span>
            {message.documentsReferenced.join(', ')}
          </div>
        )} */}

        {!message.streaming && message.historyId && (
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
