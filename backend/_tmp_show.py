import sys, json
raw = json.load(sys.stdin)
data = raw.get("data", raw) if isinstance(raw, dict) else raw
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
print(json.dumps(data, ensure_ascii=False, indent=2)[:limit])
