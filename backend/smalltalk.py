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
    "greeting": [
        "Hi there! I'm AuditAI, your audit knowledge assistant. Ask me anything about audit standards, procedures, or the documents in your knowledge base.",
        "Hello! Ready to help with any audit questions you have. What would you like to explore today?",
        "Hey! Good to see you. What audit topic can I help you with?",
    ],
    "how_are_you": [
        "I'm doing well, thank you for asking! Ready to dive into any audit questions you have. What's on your mind?",
        "All good here — running smoothly and ready to assist. What audit topic can I help with today?",
        "I'm great, thanks! Let's get to work. What would you like to look into?",
    ],
    "thanks": [
        "You're welcome! Let me know if you'd like to dig into anything else.",
        "Happy to help. Any other audit questions on your plate?",
        "Anytime. Ping me whenever something else comes up.",
    ],
    "ok": [
        "Great. Anything else I can help with?",
        "Sounds good. Let me know what's next.",
        "Got it. What else can I look into for you?",
    ],
    "bye": [
        "Goodbye! Come back any time you have audit questions.",
        "See you later. I'll be here whenever you need a hand.",
        "Take care! Looking forward to our next session.",
    ],
    "who_are_you": [
        "I'm **AuditAI** — an AI assistant built specifically for audit professionals. I answer questions using the documents in your firm's knowledge base, supplemented with general audit expertise where helpful. Think of me as a colleague who has read every audit standard, regulation, and policy your firm has uploaded.",
    ],
    "what_can_you_do": [
        "Here's what I can help you with:\n\n"
        "- **Look up audit standards** like ISA, GAAS, PCAOB pronouncements\n"
        "- **Explain audit concepts** such as materiality, sampling, going concern, and fraud risk\n"
        "- **Walk through audit procedures** for specific accounts or assertions\n"
        "- **Answer questions about firm documents** you've uploaded to the knowledge base\n"
        "- **Provide context and examples** when a topic is unclear\n\n"
        "Try asking me something like *\"What is materiality?\"* or *\"How do I audit accounts receivable?\"*"
    ],
    "compliment": [
        "Thank you, that's kind of you to say! Happy to keep helping.",
        "Glad I could help! What else would you like to look into?",
    ],
    "apology": [
        "No need to apologise! What can I help you with?",
        "All good — ask away whenever you're ready.",
    ],
    "yes": [
        "Got it. What would you like to explore next?",
        "Sounds good — let me know what you'd like to look at.",
    ],
    "no": [
        "Understood. Let me know if there's something else I can help with.",
        "No problem. I'm here whenever you need.",
    ],
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
            r"good\s+(morning|afternoon|evening|day))[\s\.\!\?\,]*$",
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
            r"appreciate\s+(it|that))[\s\.\!\?\,]*$",
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
            r"cya|later|talk\s+(to\s+you\s+)?soon|good\s+night)[\s\.\!\?\,]*$",
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


def reply_for(category: str) -> str:
    """Pick a random reply from the named category."""
    pool = _REPLIES.get(category) or _REPLIES["greeting"]
    return random.choice(pool)


def is_conversational(question: str) -> bool:
    return detect(question) is not None
