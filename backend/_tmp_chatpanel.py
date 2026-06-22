#!/usr/bin/env python3
"""Wire file-grounded chat mode into ChatPanel.js. Dry-run prints counts; 'apply' writes."""
import re, sys

P = "/Users/susovon/live-project/1audit/1audit-fe-v3/src/component/auditai/ChatPanel.js"
s = open(P, encoding="utf-8").read()
checks = {}

def repl_once(old, new, key):
    global s
    checks[key] = s.count(old)
    if s.count(old) == 1:
        s = s.replace(old, new, 1)

# 1. import chatAboutFile
repl_once(
    'import { askQuestionStream, getSessionHistory, resetSession } from "../../services/auditai.service";',
    'import { askQuestionStream, chatAboutFile, getSessionHistory, resetSession } from "../../services/auditai.service";',
    "import",
)

# 2. source-label map after STREAM_FLUSH_MS
SRC = (
    "const STREAM_FLUSH_MS = 50;\n"
    "\n// Map copilot tool names → friendly source labels shown under a file answer.\n"
    "const FILE_SOURCE_LABELS = {\n"
    '\tget_financial_statement: "financial statements",\n'
    '\tget_trial_balance: "trial balance",\n'
    '\tget_working_paper: "working paper",\n'
    '\tlist_working_papers: "working-paper index",\n'
    '\tget_risks: "risks",\n'
    '\tget_audit_area: "audit area",\n'
    '\tget_procedure_results: "procedure results",\n'
    '\tget_audit_file_summary: "file summary",\n'
    "};\n"
    "const friendlyFileSource = (name) => FILE_SOURCE_LABELS[name] || null;\n"
)
repl_once("const STREAM_FLUSH_MS = 50;\n", SRC, "source_labels")

# 3. signature: add auditFileId prop
repl_once(
    "const ChatPanel = ({ open, onClose, language, userInfo }) => {",
    "const ChatPanel = ({ open, onClose, language, userInfo, auditFileId }) => {",
    "signature",
)

# 4. mode state before streamRef
repl_once(
    "const streamRef = useRef(null);",
    'const [mode, setMode] = useState("file"); // "file" = this audit file | "kb" = standards\n\tconst streamRef = useRef(null);',
    "mode_state",
)

# 5. fileMode before the early return
repl_once(
    "if (!open) return null;",
    'const fileMode = !!auditFileId && mode === "file";\n\n\tif (!open) return null;',
    "file_mode",
)

# 6. replace the whole handleSend with a version that branches to chatAboutFile
NEW_HANDLE = (
    "const handleSend = useCallback(async (question) => {\n"
    "\t\tif (isWaiting) return;\n"
    "\t\tconst now = Date.now();\n"
    '\t\tconst userMsg = { id: `u-${now}`, role: "user", content: question, createdAt: now };\n'
    "\t\tconst aiId = `a-${now}`;\n"
    "\t\tconst aiMsg = {\n"
    "\t\t\tid: aiId,\n"
    '\t\t\trole: "assistant",\n'
    '\t\t\tcontent: "",\n'
    "\t\t\tstreaming: true,\n"
    "\t\t\tdocumentsReferenced: [],\n"
    "\t\t\tcreatedAt: now,\n"
    "\t\t\tsourceQuestion: question,\n"
    "\t\t};\n"
    "\t\tsetMessages((prev) => [...prev, userMsg, aiMsg]);\n"
    "\t\tsetIsWaiting(true);\n"
    "\n"
    "\t\t// File-grounded chat (tool-loop) is non-streaming — await the full answer.\n"
    '\t\tif (!!auditFileId && mode === "file") {\n'
    "\t\t\ttry {\n"
    "\t\t\t\tconst { answer, sources } = await chatAboutFile(question, { language, userInfo, auditFileId });\n"
    "\t\t\t\tupdateMessage(aiId, {\n"
    '\t\t\t\t\tcontent: answer || "",\n'
    "\t\t\t\t\tstreaming: false,\n"
    "\t\t\t\t\tdocumentsReferenced: (sources || []).map(friendlyFileSource).filter(Boolean),\n"
    "\t\t\t\t});\n"
    "\t\t\t} catch (e) {\n"
    "\t\t\t\tupdateMessage(aiId, {\n"
    "\t\t\t\t\tstreaming: false,\n"
    '\t\t\t\t\terror: (e && e.message) || t("There was a problem generating a response. Please try again."),\n'
    "\t\t\t\t});\n"
    "\t\t\t} finally {\n"
    "\t\t\t\tsetIsWaiting(false);\n"
    "\t\t\t}\n"
    "\t\t\treturn;\n"
    "\t\t}\n"
    "\n"
    "\t\tstreamRef.current = await askQuestionStream(question, {\n"
    "\t\t\tlanguage,\n"
    "\t\t\tuserInfo,\n"
    "\t\t\tonMeta: (m) => updateMessage(aiId, { documentsReferenced: m.documents || [] }),\n"
    "\t\t\tonDelta: (text) => queueDelta(aiId, text),\n"
    "\t\t\tonDone: (d) => {\n"
    "\t\t\t\tfinalizeStream();\n"
    "\t\t\t\tupdateMessage(aiId, {\n"
    "\t\t\t\t\tstreaming: false,\n"
    "\t\t\t\t\thistoryId: d.history_id,\n"
    "\t\t\t\t\tdocumentsReferenced: d.documents || [],\n"
    "\t\t\t\t\twasAnswered: d.was_answered,\n"
    "\t\t\t\t});\n"
    "\t\t\t\tsetIsWaiting(false);\n"
    "\t\t\t},\n"
    "\t\t\tonError: (err) => {\n"
    "\t\t\t\tfinalizeStream();\n"
    "\t\t\t\tupdateMessage(aiId, {\n"
    "\t\t\t\t\tstreaming: false,\n"
    '\t\t\t\t\terror: err || t("There was a problem generating a response. Please try again."),\n'
    "\t\t\t\t});\n"
    "\t\t\t\tsetIsWaiting(false);\n"
    "\t\t\t},\n"
    "\t\t});\n"
    "\t}, [isWaiting, language, userInfo, updateMessage, queueDelta, finalizeStream, t, auditFileId, mode]);"
)
pat6 = re.compile(
    r"const handleSend = useCallback\(async \(question\) => \{.*?\}, \[isWaiting, language, userInfo, updateMessage, queueDelta, finalizeStream, t\]\);",
    re.DOTALL,
)
m6 = pat6.findall(s)
checks["handleSend"] = len(m6)
if len(m6) == 1:
    s = pat6.sub(lambda _: NEW_HANDLE, s, count=1)

# 7. mode toggle bar before the Body comment
TOGGLE = (
    "§{auditFileId ? (\n"
    '§\t<div className="auditai-mode-toggle shrink-0 flex items-center gap-1.5 px-3 py-2 border-b border-gray-200 bg-white">\n'
    '§\t\t<span className="text-[.6875rem] text-gray-400 me-1">{t("Answer from")}:</span>\n'
    '§\t\t<button type="button" onClick={() => setMode("file")} className={`text-xs px-2.5 py-1 rounded-full font-medium transition ${mode === "file" ? "bg-violet-600 text-white" : "text-gray-600 hover:bg-gray-100"}`}>{t("This file")}</button>\n'
    '§\t\t<button type="button" onClick={() => setMode("kb")} className={`text-xs px-2.5 py-1 rounded-full font-medium transition ${mode === "kb" ? "bg-violet-600 text-white" : "text-gray-600 hover:bg-gray-100"}`}>{t("Standards")}</button>\n'
    "§\t</div>\n"
    "§) : null}\n\n"
)
def toggle_repl(m):
    indent = m.group(1)
    return TOGGLE.replace("§", indent) + indent + m.group(2)
s, n7 = re.subn(r"([ \t]*)(\{/\* Body \*/\})", toggle_repl, s, count=1)
checks["toggle"] = n7

# 8. empty-state sub text reflects the mode
repl_once(
    '{t("Ask me anything about 1audit features.")}',
    '{fileMode ? t("Ask me anything about this audit file.") : t("Ask me anything about 1audit features.")}',
    "empty_sub",
)

print("counts (each should be 1):")
ok = all(v == 1 for v in checks.values())
for k, v in checks.items():
    print(f"  {k:14} {v}  {'OK' if v == 1 else '*** MISMATCH ***'}")

if len(sys.argv) > 1 and sys.argv[1] == "apply":
    if not ok:
        print("ABORT: not all anchors matched; nothing written.")
        sys.exit(1)
    open(P, "w", encoding="utf-8").write(s)
    print("APPLIED.")
else:
    print("dry-run only (pass 'apply').")
