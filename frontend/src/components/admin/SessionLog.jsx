import { useEffect, useState } from 'react'
import { getSessionDetail, listSessions } from '../../api.js'

export default function SessionLog() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState(null)
  const [history, setHistory] = useState([])
  const [loadingHistory, setLoadingHistory] = useState(false)

  useEffect(() => {
    listSessions()
      .then((r) => setSessions(r.data))
      .finally(() => setLoading(false))
  }, [])

  const view = async (s) => {
    setSelected(s)
    setLoadingHistory(true)
    try {
      const r = await getSessionDetail(s.id)
      setHistory(r.data)
    } finally {
      setLoadingHistory(false)
    }
  }

  return (
    <div className="grid lg:grid-cols-2 gap-4">
      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b font-semibold">Sessions</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-600">
              <tr>
                <th className="text-left px-3 py-2">Started</th>
                <th className="text-left px-3 py-2">Questions</th>
                <th className="text-left px-3 py-2">IP</th>
                <th className="text-left px-3 py-2"></th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {loading && (
                <tr><td colSpan={4} className="text-center py-6 text-gray-500">Loading…</td></tr>
              )}
              {!loading && sessions.length === 0 && (
                <tr><td colSpan={4} className="text-center py-6 text-gray-500">No sessions yet.</td></tr>
              )}
              {sessions.map((s) => (
                <tr
                  key={s.id}
                  className={`hover:bg-gray-50 cursor-pointer ${selected?.id === s.id ? 'bg-brand-50' : ''}`}
                  onClick={() => view(s)}
                >
                  <td className="px-3 py-2 text-gray-600 whitespace-nowrap">
                    {new Date(s.started_at).toLocaleString()}
                  </td>
                  <td className="px-3 py-2">{s.total_questions}</td>
                  <td className="px-3 py-2 text-gray-600">{s.ip_address || '—'}</td>
                  <td className="px-3 py-2 text-right">
                    <span className="text-xs text-brand-700">View →</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b font-semibold">
          {selected ? `Questions in this session` : 'Select a session'}
        </div>
        <div className="p-4 max-h-[600px] overflow-y-auto">
          {!selected && (
            <div className="text-sm text-gray-500">
              Click a session on the left to see its questions in order.
            </div>
          )}
          {selected && loadingHistory && <div className="text-gray-500">Loading…</div>}
          {selected && !loadingHistory && history.length === 0 && (
            <div className="text-sm text-gray-500">No questions in this session yet.</div>
          )}
          <ol className="space-y-3">
            {history.map((h, i) => (
              <li key={h.id} className="border border-gray-200 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">
                  #{i + 1} · {new Date(h.asked_at).toLocaleString()} · {h.response_time_ms} ms ·{' '}
                  {h.was_answered ? '✓ Answered' : '✗ Unanswered'}
                </div>
                <div className="text-sm text-gray-900 font-medium">{h.question}</div>
                <details className="mt-2 text-xs text-gray-600">
                  <summary className="cursor-pointer text-brand-700">Show answer</summary>
                  <pre className="whitespace-pre-wrap mt-2 text-gray-800">{h.ai_answer}</pre>
                </details>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  )
}
