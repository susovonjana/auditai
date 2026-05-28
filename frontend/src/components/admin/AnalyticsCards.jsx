import { useEffect, useState } from 'react'
import { getAnalyticsSummary, getTopTopics } from '../../api.js'

function Card({ label, value, sub }) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <div className="text-xs uppercase text-gray-500 tracking-wide">{label}</div>
      <div className="text-2xl font-semibold text-gray-900 mt-1">{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  )
}

export default function AnalyticsCards() {
  const [summary, setSummary] = useState(null)
  const [topics, setTopics] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getAnalyticsSummary(), getTopTopics(10)])
      .then(([s, t]) => {
        setSummary(s.data)
        setTopics(t.data.topics)
      })
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-gray-500">Loading analytics…</div>
  if (!summary) return <div className="text-gray-500">No data yet.</div>

  const successPct = (summary.answer_success_rate * 100).toFixed(1)

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card label="Questions Today" value={summary.total_questions_today} />
        <Card label="Last 7 Days" value={summary.total_questions_week} />
        <Card label="All-Time" value={summary.total_questions_all_time} />
        <Card label="Answer Success Rate" value={`${successPct}%`} />
        <Card
          label="Avg Response Time"
          value={`${Math.round(summary.avg_response_time_ms)} ms`}
        />
        <Card label="Unique Sessions" value={summary.unique_sessions} />
        <Card
          label="Most Active Hour"
          value={
            summary.most_active_hour === null
              ? '—'
              : `${summary.most_active_hour}:00`
          }
        />
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-4">
        <h3 className="font-semibold text-gray-900 mb-2">Top 10 Topics</h3>
        {topics.length === 0 ? (
          <div className="text-sm text-gray-500">
            Not enough data yet. Keywords will appear after more questions are asked.
          </div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {topics.map((t) => (
              <span
                key={t.keyword}
                className="px-3 py-1 rounded-full bg-brand-50 text-brand-700 text-sm border border-brand-100"
              >
                {t.keyword} <span className="text-xs text-brand-500">×{t.count}</span>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
