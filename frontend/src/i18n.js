// Lightweight i18n. Two languages: en, ar. Pick text by language code.
// Persisted in localStorage so the user's choice survives reloads.

export const LANG_KEY = 'auditai_lang'

export function getLanguage() {
  const stored = (localStorage.getItem(LANG_KEY) || '').toLowerCase()
  return stored === 'ar' ? 'ar' : 'en'
}

export function setLanguage(lang) {
  const safe = lang === 'ar' ? 'ar' : 'en'
  localStorage.setItem(LANG_KEY, safe)
  // Reflect in <html> for RTL CSS
  document.documentElement.lang = safe
  document.documentElement.dir = safe === 'ar' ? 'rtl' : 'ltr'
  // Notify subscribers
  window.dispatchEvent(new CustomEvent('auditai-lang-change', { detail: safe }))
}

export function applyInitialLanguage() {
  setLanguage(getLanguage())
}

const STRINGS = {
  en: {
    app_title: 'AuditAI Assistant',
    admin_link: 'Admin',
    placeholder: 'Ask a question about 1audit',
    welcome_title: "Hello, I'm your AuditAI Assistant.",
    welcome_subtitle:
      'Ask me anything about audit standards, procedures, and regulations.',
    section_kb: 'From your knowledge base',
    section_extra: 'Additional context',
    section_key: 'Key takeaway',
    section_followups: 'Suggested follow-up questions',
    helpful: '👍 Helpful',
    not_helpful: '👎 Not Helpful',
    sources: 'Sources:',
    lang_en: 'EN',
    lang_ar: 'AR',
    lang_aria: 'Response language',
    error_generic:
      'There was a problem generating a response. Please try again.',
  },
  ar: {
    app_title: 'مساعد AuditAI',
    admin_link: 'المسؤول',
    placeholder: 'اطرح سؤالاً حول 1audit',
    welcome_title: 'مرحباً، أنا مساعدك AuditAI.',
    welcome_subtitle:
      'اسألني عن أي معيار أو إجراء أو لائحة تدقيق.',
    section_kb: 'من قاعدة المعرفة',
    section_extra: 'سياق إضافي',
    section_key: 'الخلاصة',
    section_followups: 'أسئلة متابعة مقترحة',
    helpful: '👍 مفيد',
    not_helpful: '👎 غير مفيد',
    sources: 'المصادر:',
    lang_en: 'EN',
    lang_ar: 'AR',
    lang_aria: 'لغة الرد',
    error_generic: 'حدثت مشكلة في توليد الرد. يرجى المحاولة مرة أخرى.',
  },
}

export function t(key, lang) {
  const l = (lang || getLanguage()) === 'ar' ? 'ar' : 'en'
  return STRINGS[l][key] ?? STRINGS.en[key] ?? key
}
