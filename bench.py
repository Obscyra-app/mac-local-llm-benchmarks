#!/usr/bin/env python3
"""Smoke test for local LLM as Hermes daily driver. Usage: bench.py <name> <port>"""
import json, sys, time, urllib.request

name, port = sys.argv[1], sys.argv[2]
BASE = f"http://127.0.0.1:{port}"

def post(path, payload, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data, time.time() - t0

def chat(messages, tools=None, max_tokens=300, temperature=0.3):
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return post("/v1/chat/completions", payload)

results = {"model": name}

# wait for server health (GET)
import urllib.error
for i in range(120):
    try:
        urllib.request.urlopen(BASE + "/health", timeout=5).read(); break
    except urllib.error.HTTPError:
        break
    except Exception:
        time.sleep(2)
else:
    print(json.dumps({"model": name, "error": "server not healthy"})); sys.exit(1)

# --- Test 1: plain generation speed (code task) ---
msgs = [{"role": "system", "content": "You are a concise coding assistant. Answer with code only, no explanations."},
        {"role": "user", "content": "Write a Python function `word_freq(text: str) -> dict` that counts word frequencies case-insensitively, ignoring punctuation. Then a short docstring."}]
data, dt = chat(msgs, max_tokens=350)
u = data.get("usage", {})
gen = u.get("completion_tokens", 0)
results["gen_speed"] = {"tok_s": round(gen / dt, 1) if dt > 0 else 0, "tokens": gen, "seconds": round(dt, 1),
                        "answer_head": data["choices"][0]["message"].get("content", "")[:200]}

# --- Test 2: tool calling (Hermes-critical) ---
tools = [{
    "type": "function",
    "function": {
        "name": "read_file", "description": "Read a text file from disk",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    },
}, {
    "type": "function",
    "function": {
        "name": "run_terminal", "description": "Execute a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]},
    },
}]
msgs = [{"role": "system", "content": "You are an agent. Use tools to answer. Do not guess."},
        {"role": "user", "content": "What is in the file /tmp/todo.txt? Check it."}]
data, dt = chat(msgs, tools=tools, max_tokens=200)
m = data["choices"][0]["message"]
tc = m.get("tool_calls") or []
results["tool_call"] = {
    "ok": bool(tc),
    "correct": tc and tc[0]["function"]["name"] == "read_file" and "todo.txt" in tc[0]["function"].get("arguments", ""),
    "latency_s": round(dt, 1),
    "call": (tc[0]["function"]["name"] + " " + tc[0]["function"].get("arguments", ""))[:150] if tc else m.get("content", "")[:150],
}

# --- Test 3: agent loop w/ tool result (prompt cache-ish, multi-turn) ---
msgs = [
    {"role": "system", "content": "You are a careful coding agent. Be terse. " + ("You have terminal and file tools. " * 40)},
    {"role": "user", "content": "Find the Python version on this machine."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
        "function": {"name": "run_terminal", "arguments": "{\"command\": \"python3 --version\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "Python 3.11.15"},
    {"role": "user", "content": "Now check if git is installed too."},
]
data, dt = chat(msgs, tools=tools, max_tokens=150)
m = data["choices"][0]["message"]
tc = m.get("tool_calls") or []
results["agent_loop"] = {
    "ok": bool(tc),
    "correct": tc and tc[0]["function"]["name"] == "run_terminal" and "git" in tc[0]["function"].get("arguments", ""),
    "latency_s": round(dt, 1),
    "call": (tc[0]["function"]["name"] + " " + tc[0]["function"].get("arguments", ""))[:150] if tc else m.get("content", "")[:150],
}

# --- Test 4: long-prompt prefill (~6k tokens) ---
long_ctx = "The following is an excerpt from a project log. " * 150
msgs = [{"role": "system", "content": long_ctx},
        {"role": "user", "content": "Reply with exactly one word: ACK"}]
data, dt = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 10, "temperature": 0})
u = data.get("usage", {})
pt = u.get("prompt_tokens", 0)
results["prefill"] = {"prompt_tokens": pt, "tok_s": round(pt / dt, 1) if dt > 0 else 0, "seconds": round(dt, 1)}

print(json.dumps(results, indent=2, ensure_ascii=False))
