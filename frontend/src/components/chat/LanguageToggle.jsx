import { useEffect, useState } from 'react'
import { getLanguage, setLanguage, t } from '../../i18n.js'

// A small segmented control: EN | AR. Persists choice; broadcasts change.
export default function LanguageToggle() {
  const [lang, setLang] = useState(getLanguage())

  useEffect(() => {
    const handler = (e) => setLang(e.detail)
    window.addEventListener('auditai-lang-change', handler)
    return () => window.removeEventListener('auditai-lang-change', handler)
  }, [])

  const pick = (v) => {
    if (v === lang) return
    setLanguage(v)
    setLang(v)
  }

  return (
    <div
      className="inline-flex items-center rounded-full border border-gray-200 bg-white p-0.5 text-xs"
      role="group"
      aria-label={t('lang_aria', lang)}
    >
      <button
        type="button"
        onClick={() => pick('en')}
        className={`px-2.5 py-1 rounded-full transition ${
          lang === 'en' ? 'bg-brand-600 text-white' : 'text-gray-600 hover:text-brand-700'
        }`}
        aria-pressed={lang === 'en'}
      >
        EN
      </button>
      <button
        type="button"
        onClick={() => pick('ar')}
        className={`px-2.5 py-1 rounded-full transition ${
          lang === 'ar' ? 'bg-brand-600 text-white' : 'text-gray-600 hover:text-brand-700'
        }`}
        aria-pressed={lang === 'ar'}
      >
        AR
      </button>
    </div>
  )
}
