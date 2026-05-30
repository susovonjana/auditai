import { useRef, useState } from 'react'
import { uploadDocument } from '../../api.js'

const ACCEPT = '.pdf,.xlsx,.xls,.png,.jpg,.jpeg'

const PHASES = ['Uploading', 'Extracting Text', 'Generating Embeddings', 'Active']

export default function DocumentUpload({ onUploaded }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const [category, setCategory] = useState('')
  const [items, setItems] = useState([]) // {file, percent, phase, error, doneMessage}

  const startUpload = async (file) => {
    const entry = {
      id: `${file.name}-${Date.now()}`,
      file,
      percent: 0,
      phase: PHASES[0],
      error: null,
      doneMessage: null,
    }
    setItems((p) => [entry, ...p])

    const updateItem = (id, patch) => {
      setItems((p) => p.map((e) => (e.id === id ? { ...e, ...patch } : e)))
    }

    try {
      const resp = await uploadDocument(file, category, (pct) => {
        if (pct >= 100) {
          updateItem(entry.id, { percent: 100, phase: PHASES[1] })
        } else {
          updateItem(entry.id, { percent: pct, phase: PHASES[0] })
        }
      })
      // The server runs the full pipeline before responding;
      // show the later phases briefly for UX clarity.
      updateItem(entry.id, { phase: PHASES[2] })
      setTimeout(() => {
        updateItem(entry.id, {
          phase: PHASES[3],
          doneMessage: resp.data.message,
        })
        onUploaded?.()
      }, 250)
    } catch (err) {
      updateItem(entry.id, {
        error: err?.response?.data?.detail || 'Upload failed.',
      })
    }
  }

  const onFiles = (fileList) => {
    if (!fileList?.length) return
    Array.from(fileList).forEach((f) => startUpload(f))
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    onFiles(e.dataTransfer.files)
  }

  return (
    <div className="space-y-6">
      <div className="bg-white border border-gray-200 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-1">Upload documents</h2>
        <p className="text-sm text-gray-500 mb-4">
          PDF, Word (.doc, .docx), Excel (.xlsx, .xls), and images (.png, .jpg, .jpeg). Multiple files supported.
        </p>

        <div className="flex flex-col sm:flex-row sm:items-end gap-3 mb-4">
          <div className="flex-1">
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Category (optional)
            </label>
            <input
              type="text"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              placeholder="e.g. Audit Standards"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-brand-500"
            />
          </div>
        </div>

        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition ${
            dragOver
              ? 'border-brand-600 bg-brand-50'
              : 'border-gray-300 bg-gray-50 hover:border-brand-500 hover:bg-brand-50'
          }`}
        >
          <div className="text-3xl mb-2">⬆️</div>
          <p className="text-gray-700 font-medium">
            Drag and drop files here, or click to browse
          </p>
          <p className="text-xs text-gray-500 mt-1">Maximum 25 MB per file</p>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            multiple
            className="hidden"
            onChange={(e) => onFiles(e.target.files)}
          />
        </div>
      </div>

      {items.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-xl divide-y">
          {items.map((it) => (
            <div key={it.id} className="p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-sm font-medium text-gray-900 truncate">
                    {it.file.name}
                  </div>
                  <div className="text-xs text-gray-500">
                    {(it.file.size / 1024).toFixed(1)} KB
                  </div>
                </div>
                <div className="text-xs text-gray-600 shrink-0">
                  {it.error ? (
                    <span className="text-red-600">{it.error}</span>
                  ) : it.doneMessage ? (
                    <span className="text-green-700">{it.doneMessage}</span>
                  ) : (
                    <span>
                      {it.phase} — {it.percent}%
                    </span>
                  )}
                </div>
              </div>
              {!it.error && !it.doneMessage && (
                <div className="mt-2 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brand-500 transition-all"
                    style={{ width: `${it.percent}%` }}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
