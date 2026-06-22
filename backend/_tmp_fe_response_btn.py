#!/usr/bin/env python3
"""Wire the AI 'draft response' button into ProcedureResponses (single-row
long_text layouts). Dry-run prints match counts; pass 'apply' to write."""
import re, sys

PATH = "/Users/susovon/live-project/1audit/1audit-fe-v3/src/component/files/workingPaperDetails/ProgramChecklist/ProgramChecklistProcedure.js"
s = open(PATH, encoding="utf-8").read()
counts = {}

REF_SINGLE = 'ref={(el) => { if (el) responseEditorRefs.current[`long_text_response${response.response_set_id}${sectionId}${rsp_index}`] = el; }}'
BTN_SINGLE = '{renderAiResponseBtn(`long_text_response${response.response_set_id}${sectionId}${rsp_index}`, response.response_set_id, 0)}'

block_lines = [
    "",
    '\t// --- AI "draft response" on a long_text answer, grounded in the file\'s real data ---',
    "\tconst { audit_file_id: ai_audit_file_id } = useParams();",
    "\tconst [respondingId, setRespondingId] = useState(null);",
    "\tconst responseEditorRefs = useRef({});",
    "",
    "\tconst handleGenerateResponse = async (editorId, responseSetId, rowIndex) => {",
    "\t\tif (!isEditable) {",
    '\t\t\tshowToast(t("You do not have permission to edit this section."), "error");',
    "\t\t\treturn;",
    "\t\t}",
    "\t\tif (respondingId) return;",
    '\t\tconst existing = responseEditorRefs.current[editorId]?.getEditorContent?.() || "";',
    "\t\tif (isNotEmpty(editorGetOnlyTextContent(existing))) {",
    "\t\t\tconst result = await showConfirmationDialog(",
    '\t\t\t\tt("Replace response?"),',
    '\t\t\t\tt("This replaces the current response with an AI-drafted answer grounded in this file\'s data. You can still edit it afterwards."),',
    "\t\t\t\tt,",
    '\t\t\t\t{ confirm_button_text: t("Yes, draft"), cancel_button_text: t("Cancel") },',
    "\t\t\t);",
    "\t\t\tif (!result?.isConfirmed) return;",
    "\t\t}",
    "\t\tsetRespondingId(editorId);",
    "\t\ttry {",
    "\t\t\tconst answer = await auditaiService.respondToProcedure(",
    '\t\t\t\t{ audit_file_id: Number(ai_audit_file_id), procedure: procedureText || "" },',
    '\t\t\t\t{ language: language === "ar" ? "ar" : "en", userInfo: CURRENT_USER_FROM_REDUX },',
    "\t\t\t);",
    "\t\t\tif (isNotEmpty(answer)) {",
    "\t\t\t\tresponseEditorRefs.current[editorId]?.setEditorContent?.(answer);",
    "\t\t\t\thandleCommonInputUpdate(rowIndex, responseSetId, { content_replaced: answer }, CHECKLIST_SECTION_RESPONSE_SET_TYPES.LONG_TEXT);",
    "\t\t\t} else {",
    '\t\t\t\tshowToast(t("No answer was generated. Please retry."), "error");',
    "\t\t\t}",
    "\t\t} catch (e) {",
    '\t\t\tshowToast(e?.message || t("AI generation failed. Please try again."), "error");',
    "\t\t} finally {",
    "\t\t\tsetRespondingId(null);",
    "\t\t}",
    "\t};",
    "",
    "\tconst renderAiResponseBtn = (editorId, responseSetId, rowIndex) =>",
    "\t\tisEditable ? (",
    "\t\t\t<button",
    '\t\t\t\ttype="button"',
    '\t\t\t\ttitle={respondingId === editorId ? t("Drafting…") : t("AI: answer this from the file\'s data")}',
    "\t\t\t\tdisabled={!!respondingId}",
    "\t\t\t\tonClick={(e) => {",
    "\t\t\t\t\te.stopPropagation();",
    "\t\t\t\t\thandleGenerateResponse(editorId, responseSetId, rowIndex);",
    "\t\t\t\t}}",
    '\t\t\t\tclassName="absolute top-1 end-1 z-30 flex items-center gap-1 rounded border border-violet-300 bg-white/90 px-1.5 py-0.5 text-xs font-medium text-violet-600 shadow-sm hover:bg-violet-50 disabled:cursor-not-allowed disabled:opacity-60"',
    "\t\t\t>",
    '\t\t\t\t<HiSparkles size={14} className={respondingId === editorId ? "animate-pulse" : ""} />',
    '\t\t\t\t<span className="hidden sm:inline">{respondingId === editorId ? t("Drafting…") : t("AI")}</span>',
    "\t\t\t</button>",
    "\t\t) : null;",
    "",
]
INSERT = "\n".join(block_lines) + "\n"

# 1. destructure: add CURRENT_USER_FROM_REDUX + procedureText before isEditable
s, n = re.subn(
    r'([ \t]*)SECONDARY_LANGUAGE,\n([ \t]*)isEditable,\n([ \t]*)\}\) \{',
    lambda m: m.group(1) + "SECONDARY_LANGUAGE,\n" + m.group(1) + "CURRENT_USER_FROM_REDUX,\n"
    + m.group(1) + "procedureText,\n" + m.group(2) + "isEditable,\n" + m.group(3) + "}) {",
    s)
counts["destructure"] = n

# 2. hooks + handler + button-helper, right after response_sets line
s, n = re.subn(
    r'(const response_sets = settings\?\.response_sets \?\? \[\];\n)',
    lambda m: m.group(1) + INSERT, s, count=1)
counts["hooks_block"] = n

# 3. single-row long_text editor ref (both branches: grow-relative + my-2 relative)
s, n = re.subn(
    r'\n([ \t]*)(isAutoFocus=\{activeTipTapEditorId === `long_text_response\$\{response\.response_set_id\}\$\{sectionId\}\$\{rsp_index\}` \? true : false\})',
    lambda m: "\n" + m.group(1) + REF_SINGLE + "\n" + m.group(1) + m.group(2), s)
counts["single_ref(x2)"] = n

# 4. branch-1 button (showResponsesBelow: <section className="grow relative">)
s, n = re.subn(
    r'([ \t]*)(<section\n[ \t]*className="grow relative"\n[ \t]*onClick=\{\(\) => \{\n[ \t]*if \(!isEditable\) return;\n[ \t]*onActiveTipTapEditor\(`long_text_response\$\{response\.response_set_id\}\$\{sectionId\}\$\{rsp_index\}`\);\n[ \t]*\}\}\n[ \t]*>\n)',
    lambda m: m.group(1) + m.group(2) + m.group(1) + "\t" + BTN_SINGLE + "\n", s)
counts["btn_below"] = n

# 5. branch-2 button (inline: <div className="my-2 relative">)
s, n = re.subn(
    r'([ \t]*)(<div\n[ \t]*className="my-2 relative"\n[ \t]*key=\{response\.response_set_id\}\n[ \t]*onClick=\{\(\) => \{\n[ \t]*if \(!isEditable\) return;\n[ \t]*onActiveTipTapEditor\(`long_text_response\$\{response\.response_set_id\}\$\{sectionId\}\$\{rsp_index\}`\);\n[ \t]*\}\}\n[ \t]*>\n)',
    lambda m: m.group(1) + m.group(2) + m.group(1) + "\t" + BTN_SINGLE + "\n", s)
counts["btn_inline"] = n

# 6. config call sites: add procedureText before editResponse={section_details.response}
s, n = re.subn(
    r'([ \t]*)editResponse=\{section_details\.response\}\n',
    lambda m: m.group(1) + 'procedureText={editorGetOnlyTextContent(section_details?.procedure || "")}\n'
    + m.group(1) + "editResponse={section_details.response}\n", s)
counts["config_callsites(x2)"] = n

expected = {"destructure": 1, "hooks_block": 1, "single_ref(x2)": 2, "btn_below": 1, "btn_inline": 1, "config_callsites(x2)": 2}
print("match counts:")
ok = True
for k, exp in expected.items():
    got = counts.get(k, 0)
    flag = "OK" if got == exp else "*** MISMATCH ***"
    if got != exp:
        ok = False
    print(f"  {k:24} expected {exp}  got {got}  {flag}")

if len(sys.argv) > 1 and sys.argv[1] == "apply":
    if not ok:
        print("ABORT: not all anchors matched; nothing written.")
        sys.exit(1)
    open(PATH, "w", encoding="utf-8").write(s)
    print("APPLIED.")
else:
    print("dry-run only (pass 'apply' to write).")
