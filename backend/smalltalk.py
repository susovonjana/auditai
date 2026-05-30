"""
Small-talk / conversational shortcut.

Some user messages — "hi", "thanks", "who are you" — don't benefit from
running the full RAG pipeline (embedding + retrieval + reranker + LLM).
This module detects those messages and provides instant canned replies,
bypassing the knowledge base entirely.

Detection is intentionally conservative: anything that looks like a real
question falls through to the normal Q&A pipeline.
"""
from __future__ import annotations

import re
import random
from typing import Optional


# ---------------------------------------------------------------------------
# Canned reply pools — picking randomly keeps the bot from sounding robotic.
# ---------------------------------------------------------------------------
_REPLIES = {
    "en": {
        "greeting": [
            "Hi there! I'm AuditAI, your audit knowledge assistant. Ask me anything about audit standards, procedures, or the documents in your knowledge base.",
            "Hello! Ready to help with any audit questions you have. What would you like to explore today?",
            "Hey! Good to see you. What audit topic can I help you with?",
        ],
        "how_are_you": [
            "I'm doing well, thank you for asking! Ready to dive into any audit questions you have. What's on your mind?",
            "All good here — running smoothly and ready to assist. What audit topic can I help with today?",
        ],
        "thanks": [
            "You're welcome! Let me know if you'd like to dig into anything else.",
            "Happy to help. Any other audit questions on your plate?",
        ],
        "ok": [
            "Great. Anything else I can help with?",
            "Sounds good. Let me know what's next.",
        ],
        "bye": [
            "Goodbye! Come back any time you have audit questions.",
            "See you later. I'll be here whenever you need a hand.",
        ],
        "who_are_you": [
            "I'm **AuditAI** — an AI assistant built specifically for audit professionals. I answer questions using the documents in your firm's knowledge base. Think of me as a colleague who has read every audit standard, regulation, and policy your firm has uploaded.",
        ],
        "what_can_you_do": [
            "Here's what I can help you with:\n\n"
            "- **Look up audit standards** like ISA, GAAS, PCAOB pronouncements\n"
            "- **Explain audit concepts** such as materiality, sampling, going concern, and fraud risk\n"
            "- **Walk through audit procedures** for specific accounts or assertions\n"
            "- **Answer questions about firm documents** you've uploaded to the knowledge base\n\n"
            "Try asking me something like *\"What is materiality?\"* or *\"How do I audit accounts receivable?\"*"
        ],
        "compliment": ["Thank you, that's kind of you to say! Happy to keep helping."],
        "apology": ["No need to apologise! What can I help you with?"],
        "yes": ["Got it. What would you like to explore next?"],
        "no": ["Understood. Let me know if there's something else I can help with."],
    },
    "ar": {
        "greeting": [
            "مرحباً! أنا AuditAI، مساعدك في معرفة التدقيق. اسألني عن أي معيار أو إجراء تدقيق أو أي مستند في قاعدة المعرفة لديك.",
            "أهلاً! جاهز لمساعدتك بأي سؤال متعلق بالتدقيق. ما الذي تريد استكشافه اليوم؟",
            "مرحباً بك! ما الموضوع التدقيقي الذي يمكنني مساعدتك فيه؟",
        ],
        "how_are_you": [
            "بخير، شكراً لسؤالك! جاهز للإجابة عن أي سؤال متعلق بالتدقيق. ما الذي يشغل بالك؟",
            "كل شيء جيد هنا — جاهز لمساعدتك. أي موضوع تدقيق يمكنني مناقشته معك اليوم؟",
        ],
        "thanks": [
            "العفو! أخبرني إذا أردت التعمق في أي موضوع آخر.",
            "بكل سرور. هل لديك أسئلة تدقيق أخرى؟",
        ],
        "ok": [
            "ممتاز. هل هناك شيء آخر يمكنني المساعدة به؟",
            "حسناً. أخبرني بما تريد التالي.",
        ],
        "bye": [
            "إلى اللقاء! عد في أي وقت لديك أسئلة تدقيق.",
            "أراك لاحقاً. سأكون هنا متى احتجت المساعدة.",
        ],
        "who_are_you": [
            "أنا **AuditAI** — مساعد ذكاء اصطناعي مصمم خصيصاً لمحترفي التدقيق. أجيب على الأسئلة باستخدام المستندات الموجودة في قاعدة المعرفة لشركتك. اعتبرني زميلاً قرأ كل معيار تدقيق ولائحة وسياسة رفعتها شركتك.",
        ],
        "what_can_you_do": [
            "إليك ما يمكنني مساعدتك به:\n\n"
            "- **البحث في معايير التدقيق** مثل ISA و GAAS و PCAOB\n"
            "- **شرح مفاهيم التدقيق** كالأهمية النسبية والعينات والاستمرارية ومخاطر الاحتيال\n"
            "- **استعراض إجراءات التدقيق** لحسابات أو تأكيدات محددة\n"
            "- **الإجابة عن أسئلة حول مستندات الشركة** التي رفعتها لقاعدة المعرفة\n\n"
            "جرّب أن تسألني مثلاً: *\"ما هي الأهمية النسبية؟\"* أو *\"كيف أقوم بتدقيق الذمم المدينة؟\"*"
        ],
        "compliment": ["شكراً لك، هذا لطف منك! سعيد بمواصلة المساعدة."],
        "apology": ["لا داعي للاعتذار! بماذا يمكنني مساعدتك؟"],
        "yes": ["تمام. ما الذي تريد استكشافه التالي؟"],
        "no": ["مفهوم. أخبرني إذا كان هناك شيء آخر يمكنني المساعدة به."],
    },
}


# ---------------------------------------------------------------------------
# Patterns. Each list is matched (case-insensitive) against the WHOLE question
# after stripping punctuation/whitespace. We require a fairly tight match so
# that real audit questions never get routed here.
# ---------------------------------------------------------------------------
_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "greeting",
        re.compile(
            r"^(hi|hii+|hello+|hey+|yo|howdy|hai|namaste|"
            r"good\s+(morning|afternoon|evening|day)|"
            r"مرحب[اًا]|أهل[اًا]|السلام\s*عليكم|اهلا|صباح\s*الخير|مساء\s*الخير)"
            r"[\s\.\!\?\,؟]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "how_are_you",
        re.compile(
            r"^(how\s+are\s+(you|u|ya)|how\s+is\s+it\s+going|how'?s\s+it\s+going|"
            r"how\s+are\s+things|how'?s\s+life|how\s+do\s+you\s+do|"
            r"what'?s\s+up|sup|wassup|how\s+have\s+you\s+been)[\s\.\!\?\,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "thanks",
        re.compile(
            r"^(thanks?(\s+you)?(\s+(so|very)\s+much)?|ty|thx|thnx|"
            r"thank\s+(you|u)\s+(a\s+lot|so\s+much)|much\s+appreciated|"
            r"appreciate\s+(it|that)|"
            r"شكر[اًا]|شكر[اًا]\s*لك|متشكر|الله\s+يعطيك\s+العافية)"
            r"[\s\.\!\?\,؟]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "ok",
        re.compile(
            r"^(ok(ay)?|kk|cool|nice|great|awesome|alright|"
            r"got\s+it|understood|i\s+see|makes\s+sense|sounds\s+good|"
            r"perfect|excellent)[\s\.\!\?\,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "bye",
        re.compile(
            r"^(bye+|good\s*bye|see\s+(you|ya)(\s+(later|soon))?|"
            r"cya|later|talk\s+(to\s+you\s+)?soon|good\s+night|"
            r"مع\s*السلامة|إلى\s*اللقاء|الى\s*اللقاء|وداع[اًا]|تصبح\s+على\s+خير)"
            r"[\s\.\!\?\,؟]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "who_are_you",
        re.compile(
            r"^(who\s+are\s+you|who\s+r\s+u|what\s+are\s+you|"
            r"what'?s\s+your\s+name|introduce\s+yourself|"
            r"tell\s+me\s+about\s+yourself)[\s\.\!\?\,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "what_can_you_do",
        re.compile(
            r"^(what\s+can\s+you\s+do|what\s+do\s+you\s+do|"
            r"how\s+can\s+you\s+help( me)?|what\s+are\s+your\s+(capabilities|features)|"
            r"help|menu|options|what\s+are\s+you\s+(capable|able)\s+of)[\s\.\!\?\,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "compliment",
        re.compile(
            r"^(you'?re?\s+(great|awesome|amazing|smart|helpful|the\s+best)|"
            r"good\s+(job|bot|work)|nice\s+work|well\s+done)[\s\.\!\?\,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "apology",
        re.compile(
            r"^(sorry|my\s+bad|apologies)[\s\.\!\?\,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "yes",
        re.compile(
            r"^(yes|yeah|yep|yup|sure|of\s+course|absolutely|definitely)[\s\.\!\?\,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "no",
        re.compile(
            r"^(no|nope|nah|not\s+really|not\s+now)[\s\.\!\?\,]*$",
            re.IGNORECASE,
        ),
    ),
]


def detect(question: str) -> Optional[str]:
    """
    Return the small-talk category name if `question` is conversational,
    otherwise None. The match is case-insensitive and ignores trailing
    punctuation.
    """
    if not question:
        return None
    q = question.strip()
    if not q:
        return None
    # Anything longer than ~80 chars is almost certainly a real question
    if len(q) > 80:
        return None
    for name, pattern in _PATTERNS:
        if pattern.match(q):
            return name
    return None


def reply_for(category: str, language: str = "en") -> str:
    """Pick a random reply from the named category in the requested language."""
    lang_pool = _REPLIES.get(language) or _REPLIES["en"]
    pool = lang_pool.get(category) or _REPLIES["en"]["greeting"]
    return random.choice(pool)


def is_conversational(question: str) -> bool:
    return detect(question) is not None
