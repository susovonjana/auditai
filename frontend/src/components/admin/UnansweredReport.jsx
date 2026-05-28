import { useEffect, useState } from 'react'
import { getUnanswered } from '../../api.js'

export default function UnansweredReport() {
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getUnanswered({ page: 1, page_size: 100 })
      .then((r) => {
        setItems(r.data.items)
        setTotal(r.data.total)
      })
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="space-y-4">
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
        <h2 className="font-semibold text-amber-900">Unanswered Questions</h2>
        <p className="text-sm text-amber-800 mt-1">
          These are questions where AuditAI could not find an answer in the current
          knowledge base. Use this list to decide which documents to upload next.
        </p>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-4 py-2 border-b text-sm text-gray-600">
          {loading ? 'Loading…' : `${total.toLocaleString()} unanswered question(s)`}
        </div>
        <ul className="divide-y">
          {!loading && items.length === 0 && (
            <li className="px-4 py-6 text-center text-gray-500">
              Nothing here. Great — every question has been answered so far.
            </li>
          )}
          {items.map((it) => (
            <li key={it.id} className="px-4 py-3">
              <div className="text-sm text-gray-900">{it.question}</div>
              <div className="text-xs text-gray-500 mt-1">
                {new Date(it.asked_at).toLocaleString()} · {it.response_time_ms} ms
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
