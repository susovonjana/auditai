import { useEffect, useState } from 'react'
import {
  exportSearchHistory,
  getSearchHistory,
} from '../../api.js'

const PAGE_SIZE = 25

export default function SearchHistoryTable() {
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [filter, setFilter] = useState({ q: '', answered: '' })
  const [expanded, setExpanded] = useState(null)
  const [loading, setLoading] = useState(false)

  const load = async (overridePage) => {
    setLoading(true)
    try {
      const params = { page: overridePage || page, page_size: PAGE_SIZE }
      if (filter.q) params.q = filter.q
      if (filter.answered !== '') params.answered = filter.answered === 'true'
      const r = await getSearchHistory(params)
      setItems(r.data.items)
      setTotal(r.data.total)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load(1)
    setPage(1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter])

  const exportCsv = async () => {
    const r = await exportSearchHistory()
    const url = URL.createObjectURL(r.data)
    const a = document.createElement('a')
    a.href = url
    a.download = 'auditai_search_history.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 flex flex-wrap items-center gap-2 justify-between">
        <h2 className="font-semibold text-gray-900">Search History</h2>
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="text"
            placeholder="Search questions…"
            value={filter.q}
            onChange={(e) => setFilter({ ...filter, q: e.target.value })}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
          />
          <select
            value={filter.answered}
            onChange={(e) => setFilter({ ...filter, answered: e.target.value })}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
          >
            <option value="">All answers</option>
            <option value="true">Answered</option>
            <option value="false">Unanswered</option>
          </select>
          <button
            type="button"
            onClick={exportCsv}
            className="text-sm px-3 py-1.5 border border-gray-300 rounded hover:bg-gray-50"
          >
            Export CSV
          </button>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              <th className="text-left px-4 py-2 w-44">Asked at</th>
              <th className="text-left px-4 py-2">Question</th>
              <th className="text-left px-4 py-2 w-28">Answered</th>
              <th className="text-left px-4 py-2 w-28">Response</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {loading && (
              <tr>
                <td colSpan={4} className="text-center py-6 text-gray-500">Loading…</td>
              </tr>
            )}
            {!loading && items.length === 0 && (
              <tr>
                <td colSpan={4} className="text-center py-6 text-gray-500">
                  No questions match.
                </td>
              </tr>
            )}
            {items.flatMap((it) => {
              const isOpen = expanded === it.id
              const rows = [
                <tr
                  key={it.id}
                  onClick={() => setExpanded(isOpen ? null : it.id)}
                  className="hover:bg-gray-50 cursor-pointer"
                >
                  <td className="px-4 py-2 text-gray-600 whitespace-nowrap">
                    {new Date(it.asked_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-2 text-gray-900 truncate max-w-md">
                    {it.question}
                  </td>
                  <td className="px-4 py-2">
                    {it.was_answered ? (
                      <span className="text-green-700">✓ Yes</span>
                    ) : (
                      <span className="text-red-700">✗ No</span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-gray-600">{it.response_time_ms} ms</td>
                </tr>,
              ]
              if (isOpen) {
                rows.push(
                  <tr key={`${it.id}-detail`} className="bg-gray-50">
                    <td colSpan={4} className="px-4 py-3">
                      <div className="text-xs text-gray-500 mb-1">Question:</div>
                      <div className="text-sm mb-3 text-gray-800">{it.question}</div>

                      <div className="text-xs text-gray-500 mb-1">AI answer:</div>
                      <pre className="whitespace-pre-wrap text-sm text-gray-800 bg-white p-3 rounded border border-gray-200">
                        {it.ai_answer}
                      </pre>

                      <div className="mt-2 text-xs text-gray-500">
                        Feedback: {it.user_feedback || '—'} · Documents:{' '}
                        {(it.documents_referenced || []).length}
                      </div>
                    </td>
                  </tr>,
                )
              }
              return rows
            })}
          </tbody>
        </table>
      </div>
      <div className="px-4 py-3 flex items-center justify-between border-t text-sm">
        <span className="text-gray-500">
          {total.toLocaleString()} record(s)
        </span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => { const p = Math.max(1, page - 1); setPage(p); load(p) }}
            disabled={page <= 1}
            className="px-2 py-1 border rounded disabled:opacity-40"
          >
            ‹
          </button>
          <span>{page} / {totalPages}</span>
          <button
            type="button"
            onClick={() => { const p = Math.min(totalPages, page + 1); setPage(p); load(p) }}
            disabled={page >= totalPages}
            className="px-2 py-1 border rounded disabled:opacity-40"
          >
            ›
          </button>
        </div>
      </div>
    </div>
  )
}
