import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { adminLogout, getKbStatus } from '../api.js'

import DocumentUpload from '../components/admin/DocumentUpload.jsx'
import DocumentTable from '../components/admin/DocumentTable.jsx'
import SearchHistoryTable from '../components/admin/SearchHistoryTable.jsx'
import UnansweredReport from '../components/admin/UnansweredReport.jsx'
import AnalyticsCards from '../components/admin/AnalyticsCards.jsx'
import SessionLog from '../components/admin/SessionLog.jsx'

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'upload', label: 'Upload' },
  { id: 'library', label: 'Document Library' },
  { id: 'history', label: 'Search History' },
  { id: 'unanswered', label: 'Unanswered' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'sessions', label: 'Sessions' },
]

function HealthDot({ status }) {
  const color =
    status === 'green' ? 'bg-green-500' :
    status === 'yellow' ? 'bg-amber-500' : 'bg-red-500'
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} />
}

function DashboardSection({ status, onUploadJump }) {
  if (!status) {
    return <div className="text-gray-500">Loading knowledge base status...</div>
  }
  const Card = ({ label, value, sub }) => (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <div className="text-xs uppercase text-gray-500 tracking-wide">{label}</div>
      <div className="text-2xl font-semibold text-gray-900 mt-1">{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  )
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card label="Total Documents" value={status.total_documents} />
        <Card label="Total Chunks" value={status.total_chunks} />
        <Card label="Embeddings Indexed" value={status.total_embeddings} />
        <Card
          label="KB Health"
          value={
            <span className="inline-flex items-center gap-2 text-base">
              <HealthDot status={status.health_status} />
              {status.health_status.toUpperCase()}
            </span>
          }
        />
      </div>
      <div className="bg-white border border-gray-200 rounded-xl p-4">
        <div className="text-xs uppercase text-gray-500 tracking-wide">
          Last Document Uploaded
        </div>
        <div className="mt-1 text-gray-900">
          {status.last_uploaded_filename || '—'}
        </div>
        <div className="text-xs text-gray-500 mt-1">
          {status.last_uploaded_at
            ? new Date(status.last_uploaded_at).toLocaleString()
            : 'No documents yet'}
        </div>
        <button
          type="button"
          onClick={onUploadJump}
          className="mt-3 text-sm text-brand-700 hover:underline"
        >
          + Upload a new document
        </button>
      </div>
    </div>
  )
}

export default function AdminDashboard() {
  const navigate = useNavigate()
  const [tab, setTab] = useState('dashboard')
  const [status, setStatus] = useState(null)

  const reloadStatus = () => {
    getKbStatus().then((r) => setStatus(r.data)).catch(() => {})
  }
  useEffect(reloadStatus, [])

  const handleLogout = async () => {
    try {
      await adminLogout()
    } catch (_) {}
    localStorage.removeItem('auditai_admin_token')
    localStorage.removeItem('auditai_admin_username')
    navigate('/admin/login')
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span>⚖️</span>
            <span className="font-semibold text-gray-900">AuditAI Admin</span>
            {status && (
              <span className="ml-3 inline-flex items-center gap-1.5 text-xs text-gray-500">
                <HealthDot status={status.health_status} />
                Knowledge base
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 text-sm">
            <Link to="/" className="text-gray-500 hover:text-brand-700">
              View user chat
            </Link>
            <button
              type="button"
              onClick={handleLogout}
              className="px-3 py-1.5 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
            >
              Logout
            </button>
          </div>
        </div>
        <nav className="max-w-7xl mx-auto px-4 flex gap-1 overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`px-3 py-2 text-sm border-b-2 transition whitespace-nowrap ${
                tab === t.id
                  ? 'border-brand-600 text-brand-700 font-medium'
                  : 'border-transparent text-gray-500 hover:text-gray-800'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {tab === 'dashboard' && (
          <DashboardSection
            status={status}
            onUploadJump={() => setTab('upload')}
          />
        )}
        {tab === 'upload' && (
          <DocumentUpload
            onUploaded={() => {
              reloadStatus()
              setTab('library')
            }}
          />
        )}
        {tab === 'library' && <DocumentTable onChange={reloadStatus} />}
        {tab === 'history' && <SearchHistoryTable />}
        {tab === 'unanswered' && <UnansweredReport />}
        {tab === 'analytics' && <AnalyticsCards />}
        {tab === 'sessions' && <SessionLog />}
      </main>
    </div>
  )
}
