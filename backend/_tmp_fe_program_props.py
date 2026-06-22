#!/usr/bin/env python3
"""Add procedureText prop to the two <ProcedureResponses> usages in Program.js.
Dry-run prints count; pass 'apply' to write."""
import re, sys

PATH = "/Users/susovon/live-project/1audit/1audit-fe-v3/src/component/files/workingPaperDetails/Program.js"
s = open(PATH, encoding="utf-8").read()

s2, n = re.subn(
    r'([ \t]*)editResponse=\{section\?\.section_details\?\.response\}\n',
    lambda m: m.group(1) + 'procedureText={editorGetOnlyTextContent(section?.section_details?.procedure || "")}\n'
    + m.group(1) + "editResponse={section?.section_details?.response}\n", s)

print(f"procedureText prop inserted at {n} call site(s) (expected 2)")
if len(sys.argv) > 1 and sys.argv[1] == "apply":
    if n != 2:
        print("ABORT: expected 2; nothing written.")
        sys.exit(1)
    open(PATH, "w", encoding="utf-8").write(s2)
    print("APPLIED.")
else:
    print("dry-run only.")
