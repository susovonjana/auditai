import { useEffect, useState } from 'react'
import { getLanguage, t } from '../../i18n.js'

const SUGGESTIONS = {
  en: [
    'What are the key steps in a financial statement audit?',
    'How to create a Branch in a 1audit',
    'How to create a client in 1audit',
    'What is the COA (Chart of Accounts) and how to set it up in 1audit?',
  ],
  ar: [
    'ما هي الخطوات الرئيسية في تدقيق القوائم المالية؟',
    'كيف يتم إنشاء فرع في 1audit؟',
    'كيف يتم إنشاء عميل في 1audit؟',
    'ما هو دليل الحسابات (COA) وكيف يتم إعداده في 1audit؟',
  ],
}

export default function SuggestedQuestions({ onSelect }) {
  const [lang, setLang] = useState(getLanguage())
  useEffect(() => {
    const h = (e) => setLang(e.detail)
    window.addEventListener('auditai-lang-change', h)
    return () => window.removeEventListener('auditai-lang-change', h)
  }, [])

  const suggestions = SUGGESTIONS[lang] || SUGGESTIONS.en

  return (
    <div className="max-w-3xl w-full mx-auto px-6 text-center">
      <div className="mb-6">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-brand-100 text-brand-700 text-2xl mb-3">
          ⚖️
        </div>
        <h1 className="text-2xl font-semibold text-gray-900">
          {t('welcome_title', lang)}
        </h1>
        <p className="mt-2 text-gray-600">{t('welcome_subtitle', lang)}</p>
      </div>

      <div className="grid sm:grid-cols-2 gap-3">
        {suggestions.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onSelect(q)}
            className="text-left bg-white border border-gray-200 rounded-xl px-4 py-3 hover:border-brand-500 hover:bg-brand-50 transition text-sm text-gray-700"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}
