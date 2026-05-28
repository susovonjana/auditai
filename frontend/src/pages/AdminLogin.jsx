import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { adminLogin } from '../api.js'

export default function AdminLogin() {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (params.get('expired') === '1') {
      setInfo('Your session has expired. Please log in again.')
    }
  }, [params])

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    setInfo('')
    setSubmitting(true)
    try {
      const r = await adminLogin(username, password)
      localStorage.setItem('auditai_admin_token', r.data.access_token)
      localStorage.setItem('auditai_admin_username', r.data.username)
      navigate('/admin', { replace: true })
    } catch (err) {
      setError(
        err?.response?.data?.detail || 'Incorrect username or password.',
      )
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 px-4">
      <div className="w-full max-w-sm bg-white border border-gray-200 rounded-2xl shadow-sm p-8">
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-brand-100 text-brand-700 text-xl mb-2">
            🔐
          </div>
          <h1 className="text-xl font-semibold text-gray-900">AuditAI Admin</h1>
          <p className="text-sm text-gray-500 mt-1">
            Sign in to manage the knowledge base
          </p>
        </div>

        {info && (
          <div className="mb-4 p-3 bg-amber-50 border border-amber-200 text-amber-800 text-sm rounded-lg">
            {info}
          </div>
        )}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg">
            {error}
          </div>
        )}

        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Username
            </label>
            <input
              type="text"
              autoFocus
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
            />
          </div>
          <button
            type="submit"
            disabled={submitting}
            className="w-full py-2.5 bg-brand-600 text-white rounded-lg hover:bg-brand-700 disabled:bg-gray-300 transition font-medium"
          >
            {submitting ? 'Signing in...' : 'Sign in'}
          </button>
        </form>

        <div className="mt-6 text-center">
          <Link to="/" className="text-xs text-gray-500 hover:text-brand-700">
            ← Back to chat
          </Link>
        </div>
      </div>
    </div>
  )
}
