import { useEffect, useState } from 'react'
import { deleteDocument, listDocuments, updateDocument } from '../../api.js'

const ICONS = { pdf: '📄', xlsx: '📊', xls: '📊', png: '🖼️', jpg: '🖼️', jpeg: '🖼️' }

function StatusBadge({ status }) {
  const map = {
    active: 'bg-green-100 text-green-700',
    processing: 'bg-amber-100 text-amber-700',
    error: 'bg-red-100 text-red-700',
  }
  return (
    <span
      className={`px-2 py-0.5 rounded-full text-xs font-medium ${map[status] || 'bg-gray-100 text-gray-600'}`}
    >
      {status}
    </span>
  )
}

export default function DocumentTable({ onChange }) {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(true)
  const [editingId, setEditingId] = useState(null)
  const [editValue, setEditValue] = useState('')

  const load = async () => {
    setLoading(true)
    try {
      const r = await listDocuments()
      setDocs(r.data)
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { load() }, [])

  const saveCategory = async (doc) => {
    await updateDocument(doc.id, { category: editValue })
    setEditingId(null)
    load()
  }

  const remove = async (doc) => {
    if (!window.confirm(`Delete "${doc.filename}" and all its embeddings?`)) return
    await deleteDocument(doc.id)
    load()
    onChange?.()
  }

  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
        <h2 className="font-semibold text-gray-900">Document Library</h2>
        <button
          type="button"
          onClick={load}
          className="text-sm text-brand-700 hover:underline"
        >
          Refresh
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              <th className="text-left px-4 py-2">File</th>
              <th className="text-left px-4 py-2">Category</th>
              <th className="text-left px-4 py-2">Uploaded</th>
              <th className="text-left px-4 py-2">Chunks</th>
              <th className="text-left px-4 py-2">Status</th>
              <th className="text-right px-4 py-2">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {loading && (
              <tr>
                <td colSpan={6} className="text-center py-6 text-gray-500">
                  Loading…
                </td>
              </tr>
            )}
            {!loading && docs.length === 0 && (
              <tr>
                <td colSpan={6} className="text-center py-6 text-gray-500">
                  No documents yet. Upload your first document to populate the knowledge base.
                </td>
              </tr>
            )}
            {docs.map((d) => (
              <tr key={d.id} className="hover:bg-gray-50">
                <td className="px-4 py-3">
                  <span className="mr-2">{ICONS[d.file_type] || '📁'}</span>
                  <span className="font-medium text-gray-900">{d.filename}</span>
                </td>
                <td className="px-4 py-3">
                  {editingId === d.id ? (
                    <div className="flex gap-1">
                      <input
                        type="text"
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        className="border border-gray-300 rounded px-2 py-1 text-sm w-40"
                      />
                      <button
                        type="button"
                        onClick={() => saveCategory(d)}
                        className="text-xs px-2 py-1 bg-brand-600 text-white rounded"
                      >
                        Save
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => {
                        setEditingId(d.id)
                        setEditValue(d.category || '')
                      }}
                      className="text-gray-700 hover:text-brand-700"
                    >
                      {d.category || <span className="text-gray-400 italic">+ Add</span>}
                    </button>
                  )}
                </td>
                <td className="px-4 py-3 text-gray-600">
                  {new Date(d.uploaded_at).toLocaleString()}
                </td>
                <td className="px-4 py-3">{d.total_chunks}</td>
                <td className="px-4 py-3"><StatusBadge status={d.status} /></td>
                <td className="px-4 py-3 text-right">
                  <button
                    type="button"
                    onClick={() => remove(d)}
                    className="text-red-600 hover:text-red-800 text-sm"
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
