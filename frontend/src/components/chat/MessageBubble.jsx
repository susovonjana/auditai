import { memo, useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { sendFeedback } from '../../api.js'
import { getLanguage, t } from '../../i18n.js'

const NO_ANSWER_PATTERNS = [
  "i'm sorry. i'm unable to help",
  'unable to help you with your query',
  'عذراً، أنا غير قادر',
  'غير قادر على مساعدتك',
]
const isNoAnswer = (content) => {
  if (!content) return false
  const c = content.toLowerCase()
  return NO_ANSWER_PATTERNS.some((p) => c.includes(p.toLowerCase()))
}

const MailIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 shrink-0">
    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
    <polyline points="22,6 12,13 2,6"></polyline>
  </svg>
)
const PhoneIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 shrink-0">
    <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
  </svg>
)

function NoAnswerCard() {
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 p-3.5">
      <div className="flex items-start gap-2 mb-3">
        <span className="text-amber-600 mt-0.5">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5 shrink-0">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
        </span>
        <div className="text-sm font-medium text-gray-800 leading-snug">
          I'm sorry, I couldn't find an answer to your query in the knowledge base.
        </div>
      </div>
      <div className="text-xs text-gray-600 mb-2.5">
        Our support team is happy to help you directly:
      </div>
      <div className="flex flex-col gap-1.5">
        <a href="mailto:info@1audit.com" className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white border border-amber-200 hover:border-amber-400 hover:bg-amber-100 transition text-sm text-gray-800 no-underline">
          <span className="text-amber-600"><MailIcon /></span>
          <span className="font-medium">info@1audit.com</span>
        </a>
        <a href="tel:+966920035129" className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white border border-amber-200 hover:border-amber-400 hover:bg-amber-100 transition text-sm text-gray-800 no-underline">
          <span className="text-amber-600"><PhoneIcon /></span>
          <span className="font-medium" dir="ltr">+966 920 035 129</span>
        </a>
      </div>
    </div>
  )
}

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

function FollowUps({ questions, onSelect, disabled, title }) {
  if (!questions || questions.length === 0) return null
  return (
    <div className="mt-3 rounded-xl border bg-indigo-50 border-indigo-100 px-4 py-3">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide mb-2 text-indigo-700">
        <span>💬</span>
        <span>{title || 'Explore Related Topics'}</span>
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

function MessageBubble({ message, onSendQuestion, onRetry, isBusy }) {
  const [feedback, setFeedback] = useState(message.feedback || null)
  const [submitting, setSubmitting] = useState(false)
  const [lang, setLang] = useState(getLanguage())
  const isUser = message.role === 'user'
  const sections = useMemo(() => splitSections(message.content), [message.content])
  const noAnswer = useMemo(() => isNoAnswer(message.content), [message.content])

  useEffect(() => {
    const h = (e) => setLang(e.detail)
    window.addEventListener('auditai-lang-change', h)
    return () => window.removeEventListener('auditai-lang-change', h)
  }, [])

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
        {message.error && (
          <div
            role="alert"
            className="rounded-xl border border-red-200 bg-red-50 p-3.5 mb-3"
          >
            <div className="flex items-start gap-2">
              <span className="text-red-600 mt-0.5">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5 shrink-0">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="8" x2="12" y2="12"></line>
                  <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
              </span>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-red-900 leading-snug">
                  {t('error_generic', lang)}
                </div>
                <div className="text-xs text-red-700 mt-1 break-words">{message.error}</div>
                {message.sourceQuestion && onRetry && (
                  <button
                    type="button"
                    onClick={() => onRetry(message.id, message.sourceQuestion)}
                    disabled={isBusy}
                    className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-red-300 text-red-700 hover:bg-red-100 transition text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-3.5 h-3.5">
                      <polyline points="1 4 1 10 7 10"></polyline>
                      <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
                    </svg>
                    {t('retry', lang)}
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
        {message.stopped && !message.error && (
          <div className="text-xs italic text-gray-500 mb-2">
            {t('stopped', lang)}
          </div>
        )}
        {noAnswer ? (
          <NoAnswerCard />
        ) : sections ? (
          <>
            {sections.kb && (
              <Section
                accent="bg-brand-50 border-brand-100"
                title={t('section_kb', lang)}
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
                title={t('section_extra', lang)}
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
                title={t('section_key', lang)}
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
              title={t('section_followups', lang)}
            />
          </>
        ) : (
          <div className="markdown text-gray-800">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content || ''}
            </ReactMarkdown>
          </div>
        )}

        {!message.error && message.documentsReferenced?.length > 0 && (
          <div className="mt-3 pt-2 border-t border-gray-100">
            <div className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1">
              {t('sources', lang)}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {message.documentsReferenced.map((src, i) => (
                <span
                  key={`${src}-${i}`}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-gray-50 border border-gray-200 text-xs text-gray-700 max-w-full"
                  title={src}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-3 h-3 shrink-0 text-gray-400">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                  </svg>
                  <span className="truncate">{src}</span>
                </span>
              ))}
            </div>
          </div>
        )}

        {!message.streaming && message.historyId && !message.error && (
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

// React.memo + custom equality: skip re-render when fields we render didn't
// change. Without this, every streamed token re-renders every bubble in the
// thread (O(n²) over n tokens × m messages).
const arePropsEqual = (prev, next) => {
  if (prev.isBusy !== next.isBusy) return false
  if (prev.onSendQuestion !== next.onSendQuestion) return false
  if (prev.onRetry !== next.onRetry) return false
  const a = prev.message
  const b = next.message
  return (
    a.id === b.id &&
    a.content === b.content &&
    a.streaming === b.streaming &&
    a.error === b.error &&
    a.stopped === b.stopped &&
    a.historyId === b.historyId &&
    (a.documentsReferenced?.length || 0) === (b.documentsReferenced?.length || 0)
  )
}

export default memo(MessageBubble, arePropsEqual)
