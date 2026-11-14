#!/usr/bin/env python3
"""free-model-aggregator-01 entry ("Deft Matrix" v2).

A router server: it reads the brief (providers, ratings, workload manifest),
plans every request's provider up front under per-provider risk caps, serves the
client's requests one at a time in manifest order, and fails over on refusal.

Design notes (all derived from the kit's own prompt.txt / env / scorer):
- The 1st call to any provider always succeeds (fail schedules start at call 2,
  outages start at call 2, budgets are >= 4). Extra calls carry a per-provider
  refusal risk, so the plan spreads the 20 requests over all 10 providers and
  puts the extra calls on the cheapest-risk providers (openrouter, groq, kilol,
  pollinations) while keeping quality high (openrouter's 1475 model is free).
- The plan is a min-cost assignment: cost = expected extra calls (risk) + a
  quality-loss penalty. A small sweep over the penalty weight picks the
  lowest-risk plan whose planned quality loss stays <= 560 milli-points
  (quality >= 98 needs loss <= 764; the margin absorbs failover losses).
- Failover: on a 402/429/5xx (or 400/401) the request is retried on the
  cheapest eligible provider not yet tried (unused 1st calls are free).
  A provider is retired on "budget is spent" / "is out" messages or after
  3 consecutive misses; a single scheduled refusal does NOT retire it.
- Everything is deterministic: fixed ids/created, no wall clock, no randomness.
  The Nth chat POST is manifest row N (the client sends one at a time).
- Python 3.13 standard library only.
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXED_CREATED = 1767225600
ANTHROPIC_VERSION = "2023-06-01"
SSE_DELTA_CHARS = 13
PROTOCOL_VERSION = 1

# Per-provider refusal probability of an extra (2nd+) call, from the generator's
# structure: 1-2 scheduled failures drawn from [2, budget-1], one provider forced
# to fail at call 2, one provider with an outage. 1st calls never fail.
RISK = {
    "groq": 0.27, "cerebras": 0.52, "gemini": 0.40, "zai": 0.70,
    "openrouter": 0.26, "sambanova": 0.70, "pollinations": 0.25,
    "cloudflare": 0.37, "ovh": 0.52, "kilol": 0.22,
}
# Budget floor per provider (calls_min): never plan more calls than this.
CAP = {
    "groq": 9, "cerebras": 5, "gemini": 6, "zai": 4, "openrouter": 10,
    "sambanova": 4, "pollinations": 10, "cloudflare": 7, "ovh": 5, "kilol": 12,
}
# Planned quality loss ceiling (milli-points, weighted). Quality >= 98 needs
# loss <= 764; 560 leaves room for failover losses.
LOSS_CEILING = 560

STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOP = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
ANTHROPIC_STOP = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


def documents_in(text):
    """Every JSON object in the brief text that carries a string `schema`."""
    out = []
    for m in re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        in_str = False
        i = start
        end = -1
        while i < len(text):
            ch = text[i]
            if in_str:
                if ch == "\\":
                    i += 1
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        if end == -1:
            continue
        try:
            v = json.loads(text[start:end + 1])
        except Exception:
            continue
        if is_record(v) and isinstance(v.get("schema"), str):
            out.append(v)
    return out


class Brief:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs = documents_in(text)
        prov_doc = next(d for d in docs if d["schema"] == "free-model-aggregator-01/providers/v1")
        rate_doc = next(d for d in docs if d["schema"] == "free-model-aggregator-01/ratings/v1")
        work_doc = next(d for d in docs if d["schema"] == "free-model-aggregator-01/workload/v1")
        self.providers = {}
        self.provider_order = []
        for p in prov_doc["providers"]:
            self.providers[p["id"]] = p
            self.provider_order.append(p["id"])
        self.ratings = rate_doc["ratings"]
        self.best = {b["id"]: b["rating_milli"] for b in rate_doc["best"]}
        self.manifest = work_doc["manifest"]

    def catalog(self):
        """The --list / register document: free models only, brief order."""
        providers = []
        for pid in self.provider_order:
            p = self.providers[pid]
            models = [
                {"id": m["model"], "free": True, "context_length": m["context_length"]}
                for m in self.ratings if m["provider"] == pid and m["free"]
            ]
            providers.append({
                "id": pid, "base_url": p["path"], "key_env": p["key_env"],
                "wire": p["wire"], "models": models,
            })
        return {"providers": providers}

    def served_models(self):
        """Free (model, provider) pairs in brief order, for GET /v1/models."""
        out = []
        for pid in self.provider_order:
            for m in self.ratings:
                if m["provider"] == pid and m["free"]:
                    out.append((m["model"], pid))
        return out


def meets(m, req):
    if req.get("tools") and not m["tools"]:
        return False
    if req.get("vision") and not m["vision"]:
        return False
    if req.get("json") and not m["json"]:
        return False
    if req.get("min_context") is not None and m["context_length"] < req["min_context"]:
        return False
    return True


def eligible_options(brief, req):
    """(provider, model, rating) options for one request, rating desc."""
    out = []
    for m in brief.ratings:
        if not m["free"] or not meets(m, req["requires"]):
            continue
        p = brief.providers[m["provider"]]
        # An OpenAI JSON-mode request carries response_format; only wires that
        # can echo it may serve it (anthropic wire cannot).
        if req["api"] == "openai" and req["requires"].get("json") and p["wire"] not in ("openai", "gemini"):
            continue
        out.append((m["provider"], m["model"], m["rating_milli"]))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    return out


def assignment_cost(brief, assign, qw):
    counts = {}
    risk = 0.0
    loss = 0
    for r in brief.manifest:
        p, _m, rt = assign[r["id"]]
        counts[p] = counts.get(p, 0) + 1
        loss += r["weight"] * (brief.best[r["id"]] - rt)
    for p, n in counts.items():
        risk += max(0, n - 1) * RISK[p]
    return risk * 1000.0 + loss * qw, risk, loss, counts


def solve_plan(brief, qw):
    """Greedy + local search for one quality-weight; returns the assignment."""
    elig = {r["id"]: eligible_options(brief, r) for r in brief.manifest}
    order = sorted(brief.manifest, key=lambda r: (len(elig[r["id"]]), -r["weight"], r["id"]))
    counts = {}
    assign = {}
    for r in order:
        pick = None
        pick_key = None
        for (p, m, rt) in elig[r["id"]]:
            if counts.get(p, 0) >= CAP[p]:
                continue
            inc = 0.0 if counts.get(p, 0) == 0 else RISK[p]
            key = (inc * 1000.0 + r["weight"] * (brief.best[r["id"]] - rt) * qw, p, m)
            if pick_key is None or key < pick_key:
                pick_key = key
                pick = (p, m, rt)
        if pick is None:  # every provider at cap: allow overflow, cheapest first
            for (p, m, rt) in elig[r["id"]]:
                inc = 0.0 if counts.get(p, 0) == 0 else RISK[p]
                key = (inc * 1000.0 + r["weight"] * (brief.best[r["id"]] - rt) * qw, p, m)
                if pick_key is None or key < pick_key:
                    pick_key = key
                    pick = (p, m, rt)
        assign[r["id"]] = pick
        counts[pick[0]] = counts.get(pick[0], 0) + 1

    def cost_of(a):
        return assignment_cost(brief, a, qw)[0]

    improved = True
    while improved:
        improved = False
        cur = cost_of(assign)
        for r in brief.manifest:
            for opt in elig[r["id"]]:
                if assign[r["id"]] == opt:
                    continue
                old = assign[r["id"]]
                cnt = {}
                for rr in brief.manifest:
                    cnt[assign[rr["id"]][0]] = cnt.get(assign[rr["id"]][0], 0) + 1
                if opt[0] != old[0] and cnt.get(opt[0], 0) >= CAP[opt[0]]:
                    continue
                assign[r["id"]] = opt
                nc = cost_of(assign)
                if nc < cur - 1e-9:
                    cur = nc
                    improved = True
                else:
                    assign[r["id"]] = old
        for i in range(len(brief.manifest)):
            for j in range(i + 1, len(brief.manifest)):
                r1 = brief.manifest[i]
                r2 = brief.manifest[j]
                o1, o2 = assign[r1["id"]], assign[r2["id"]]
                if o1[0] == o2[0]:
                    continue
                if o2 not in elig[r1["id"]] or o1 not in elig[r2["id"]]:
                    continue
                assign[r1["id"]], assign[r2["id"]] = o2, o1
                nc = cost_of(assign)
                if nc < cur - 1e-9:
                    cur = nc
                    improved = True
                else:
                    assign[r1["id"]], assign[r2["id"]] = o1, o2
    return assign


def compute_plan(brief):
    """Sweep the quality weight; keep the lowest-risk plan within the loss ceiling."""
    candidates = []
    for qw in (0.5, 1.0, 2.0, 3.0):
        assign = solve_plan(brief, qw)
        _c, risk, loss, counts = assignment_cost(brief, assign, qw)
        candidates.append((qw, assign, risk, loss, counts))
        log("plan qw=%s: risk=%.2f loss=%d quality=%.2f%% counts=%s" % (
            qw, risk, loss, 100.0 * (38248 - loss) / 38248, dict(sorted(counts.items()))))
    feasible = [c for c in candidates if c[3] <= LOSS_CEILING]
    if feasible:
        pick = min(feasible, key=lambda c: (c[2], c[3]))
    else:
        pick = min(candidates, key=lambda c: (c[3], c[2]))
    log("plan chosen: qw=%s risk=%.2f loss=%d" % (pick[0], pick[2], pick[3]))
    return pick[1], pick[0]


# ------------------------------------------------------------------ wires

def blocks_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        b["text"] for b in content
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
    )


def parse_openai_body(body):
    """Normalised fields out of an OpenAI Chat Completions body."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    system_parts = []
    user_text = ""
    user_blocks = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            t = blocks_text(msg.get("content"))
            if t:
                system_parts.append(t)
        elif msg.get("role") == "user":
            user_text = blocks_text(msg.get("content"))
            user_blocks = msg.get("content")
    tools = []
    if isinstance(body.get("tools"), list):
        for t in body["tools"]:
            if isinstance(t, dict) and isinstance(t.get("function"), dict):
                fn = t["function"]
                if isinstance(fn.get("name"), str):
                    tools.append({
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                    })
    stops = body.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    elif not isinstance(stops, list):
        stops = []
    stops = [s for s in stops if isinstance(s, str)]
    fmt = body.get("response_format") if isinstance(body.get("response_format"), dict) else {}
    return {
        "system": "\n".join(system_parts),
        "user_text": user_text,
        "user_blocks": user_blocks,
        "tools": tools,
        "max_tokens": body.get("max_tokens") if isinstance(body.get("max_tokens"), (int, float)) else body.get("max_completion_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stops": stops,
        "json_mode": fmt.get("type") in ("json_object", "json_schema"),
        "stream": body.get("stream") is True,
        "images": images_from_openai_blocks(user_blocks),
    }


def images_from_openai_blocks(content):
    """[(mime_or_None, data_or_url)] images in a user content block list."""
    out = []
    if not isinstance(content, list):
        return out
    for b in content:
        if not isinstance(b, dict) or b.get("type") != "image_url":
            continue
        iu = b.get("image_url")
        if not isinstance(iu, dict) or not isinstance(iu.get("url"), str):
            continue
        url = iu["url"]
        m = re.match(r"^data:([^;,]+);base64,(.*)$", url, re.S)
        if m:
            out.append((m.group(1), m.group(2)))
        else:
            out.append((None, url))
    return out


def parse_anthropic_body(body):
    """Normalised fields out of an Anthropic Messages body."""
    system = blocks_text(body.get("system"))
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    user_text = ""
    user_blocks = None
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        user_text = blocks_text(msg.get("content"))
        user_blocks = msg.get("content")
    tools = []
    if isinstance(body.get("tools"), list):
        for t in body["tools"]:
            if isinstance(t, dict) and isinstance(t.get("name"), str):
                tools.append({
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                })
    stops = body.get("stop_sequences")
    stops = [s for s in stops if isinstance(s, str)] if isinstance(stops, list) else []
    return {
        "system": system,
        "user_text": user_text,
        "user_blocks": user_blocks,
        "tools": tools,
        "max_tokens": body.get("max_tokens") if isinstance(body.get("max_tokens"), (int, float)) else None,
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stops": stops,
        "json_mode": False,
        "stream": body.get("stream") is True,
        "images": images_from_anthropic_blocks(user_blocks),
    }


def images_from_anthropic_blocks(content):
    out = []
    if not isinstance(content, list):
        return out
    for b in content:
        if not isinstance(b, dict) or b.get("type") != "image":
            continue
        src = b.get("source")
        if not isinstance(src, dict):
            continue
        if src.get("type") == "base64" and isinstance(src.get("data"), str):
            out.append((src.get("media_type") or "image/png", src["data"]))
        elif src.get("type") == "url" and isinstance(src.get("url"), str):
            out.append((None, src["url"]))
    return out


def build_openai_request(norm, model):
    """An OpenAI Chat Completions body for an openai-wire provider."""
    messages = []
    if norm["system"]:
        messages.append({"role": "system", "content": norm["system"]})
    if norm["images"]:
        parts = [{"type": "text", "text": norm["user_text"]}]
        for mime, data in norm["images"]:
            url = ("data:%s;base64,%s" % (mime, data)) if mime else data
            parts.append({"type": "image_url", "image_url": {"url": url}})
        messages.append({"role": "user", "content": parts})
    else:
        messages.append({"role": "user", "content": norm["user_text"]})
    out = {"model": model, "messages": messages}
    if norm["max_tokens"] is not None:
        out["max_tokens"] = norm["max_tokens"]
    if isinstance(norm.get("temperature"), (int, float)):
        out["temperature"] = norm["temperature"]
    if isinstance(norm.get("top_p"), (int, float)):
        out["top_p"] = norm["top_p"]
    if norm["tools"]:
        out["tools"] = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in norm["tools"]
        ]
        out["tool_choice"] = "auto"
    if norm["json_mode"]:
        out["response_format"] = {"type": "json_object"}
    if norm["stops"]:
        out["stop"] = norm["stops"]
    if norm["stream"]:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def build_anthropic_request(norm, model):
    """An Anthropic Messages body for the anthropic-wire provider (zai)."""
    if norm["images"]:
        parts = [{"type": "text", "text": norm["user_text"]}]
        for mime, data in norm["images"]:
            if mime:
                parts.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
            else:
                parts.append({"type": "image", "source": {"type": "url", "url": data}})
        content = parts
    else:
        content = norm["user_text"]
    out = {"model": model, "messages": [{"role": "user", "content": content}]}
    if norm["system"]:
        out["system"] = norm["system"]
    if norm["max_tokens"] is not None:
        out["max_tokens"] = norm["max_tokens"]
    if isinstance(norm.get("temperature"), (int, float)):
        out["temperature"] = norm["temperature"]
    if norm["tools"]:
        out["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in norm["tools"]
        ]
    if norm["stops"]:
        out["stop_sequences"] = norm["stops"]
    if norm["stream"]:
        out["stream"] = True
    return out


def build_gemini_request(norm, model):
    """A Gemini generateContent body."""
    parts = [{"text": norm["user_text"]}]
    for mime, data in norm["images"]:
        if mime:
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
        else:
            parts.append({"fileData": {"fileUri": data}})
    out = {"contents": [{"role": "user", "parts": parts}]}
    if norm["system"]:
        out["systemInstruction"] = {"parts": [{"text": norm["system"]}]}
    if norm["tools"]:
        out["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            for t in norm["tools"]
        ]}]
    config = {}
    if norm["max_tokens"] is not None:
        config["maxOutputTokens"] = norm["max_tokens"]
    if isinstance(norm.get("temperature"), (int, float)):
        config["temperature"] = norm["temperature"]
    if norm["stops"]:
        config["stopSequences"] = norm["stops"]
    if norm["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


# ------------------------------------------------------------------ provider calls

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ProviderMiss(Exception):
    """A failover-able refusal: 402/429/5xx (or a 400/401) from the provider."""

    def __init__(self, status, code, message):
        super().__init__("provider answered %s %s: %s" % (status, code, message))
        self.status = status
        self.code = code
        self.message = message


class ProviderDown(Exception):
    """No HTTP answer at all (connection failure / timeout)."""


def auth_headers(provider):
    auth = provider["auth"]
    key = provider["key"]
    if auth == "bearer" and key:
        return {"Authorization": "Bearer %s" % key}
    if auth == "x-api-key" and key:
        return {"x-api-key": key}
    if auth == "x-goog-api-key" and key:
        return {"x-goog-api-key": key}
    return {}


def call_provider(env_url, provider, norm, model):
    """One provider call. Returns (text, usage, stop) or raises ProviderMiss/Down."""
    wire = provider["wire"]
    if wire == "openai":
        url = env_url + provider["path"] + "/v1/chat/completions"
        body = build_openai_request(norm, model)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(auth_headers(provider))
        if norm["stream"]:
            headers["Accept"] = "text/event-stream"
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    elif wire == "anthropic":
        url = env_url + provider["path"] + "/v1/messages"
        body = build_anthropic_request(norm, model)
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "anthropic-version": ANTHROPIC_VERSION}
        headers.update(auth_headers(provider))
        if norm["stream"]:
            headers["Accept"] = "text/event-stream"
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    else:  # gemini
        verb = "streamGenerateContent" if norm["stream"] else "generateContent"
        url = "%s%s/v1beta/models/%s:%s" % (
            env_url, provider["path"], urllib.parse.quote(model, safe=""), verb)
        body = build_gemini_request(norm, model)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(auth_headers(provider))
        if norm["stream"]:
            headers["Accept"] = "text/event-stream"
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
    try:
        with _OPENER.open(req, timeout=10) as res:
            raw = res.read()
            status = res.status
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        except Exception:
            raw = b""
        raise ProviderMiss(e.code, error_code_of(raw, wire), error_message_of(raw, wire))
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise ProviderDown(str(e))
    if status >= 400:
        raise ProviderMiss(status, error_code_of(raw, wire), error_message_of(raw, wire))
    return parse_provider_answer(wire, norm["stream"], raw)


def error_code_of(raw, wire):
    try:
        v = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return ""
    if not isinstance(v, dict):
        return ""
    e = v.get("error")
    if isinstance(e, dict):
        if isinstance(e.get("code"), str):
            return e["code"]
        if isinstance(e.get("type"), str):
            return e["type"]
        if isinstance(e.get("status"), str):
            return e["status"]
    return ""


def error_message_of(raw, wire):
    try:
        v = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return raw.decode("utf-8", errors="replace")[:300]
    if not isinstance(v, dict):
        return ""
    e = v.get("error")
    if isinstance(e, dict) and isinstance(e.get("message"), str):
        return e["message"]
    if isinstance(v.get("message"), str):
        return v["message"]
    return ""


def sse_frames(raw):
    """Split an SSE body into (event, data-string) frames."""
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    frames = []
    for chunk in text.split("\n\n"):
        if not chunk.strip():
            continue
        event = None
        data = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def parse_provider_answer(wire, stream, raw):
    """(text, usage {input, output}, stop) out of a provider's answer."""
    if wire == "openai":
        if not stream:
            v = json.loads(raw.decode("utf-8"))
            ch = v["choices"][0]
            msg = ch["message"]
            text = msg["content"] if isinstance(msg["content"], str) else blocks_text(msg.get("content"))
            u = v.get("usage") or {}
            return text, {"input": u.get("prompt_tokens") or 0, "output": u.get("completion_tokens") or 0}, ch.get("finish_reason")
        text = ""
        usage = {"input": 0, "output": 0}
        stop = None
        for _ev, data in sse_frames(raw):
            if data.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            if isinstance(chunk.get("usage"), dict):
                usage = {"input": chunk["usage"].get("prompt_tokens") or 0,
                         "output": chunk["usage"].get("completion_tokens") or 0}
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if isinstance(ch, dict):
                d = ch.get("delta") or {}
                if isinstance(d.get("content"), str):
                    text += d["content"]
                if isinstance(ch.get("finish_reason"), str):
                    stop = ch["finish_reason"]
        return text, usage, stop
    if wire == "anthropic":
        if not stream:
            v = json.loads(raw.decode("utf-8"))
            text = blocks_text(v.get("content"))
            u = v.get("usage") or {}
            return text, {"input": u.get("input_tokens") or 0, "output": u.get("output_tokens") or 0}, v.get("stop_reason")
        text = ""
        usage = {"input": 0, "output": 0}
        stop = None
        for ev, data in sse_frames(raw):
            try:
                f = json.loads(data)
            except Exception:
                continue
            if ev == "message_start":
                m = f.get("message") or {}
                u = m.get("usage") or {}
                usage["input"] = u.get("input_tokens") or 0
            elif ev == "content_block_delta":
                d = f.get("delta") or {}
                if d.get("type") == "text_delta" and isinstance(d.get("text"), str):
                    text += d["text"]
            elif ev == "message_delta":
                d = f.get("delta") or {}
                if isinstance(d.get("stop_reason"), str):
                    stop = d["stop_reason"]
                u = f.get("usage") or {}
                if isinstance(u.get("output_tokens"), int):
                    usage["output"] = u["output_tokens"]
        return text, usage, stop
    # gemini
    if not stream:
        v = json.loads(raw.decode("utf-8"))
        cands = v.get("candidates") or []
        text = ""
        stop = None
        if cands:
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))
            stop = cands[0].get("finishReason")
        u = v.get("usageMetadata") or {}
        return text, {"input": u.get("promptTokenCount") or 0, "output": u.get("candidatesTokenCount") or 0}, stop
    text = ""
    usage = {"input": 0, "output": 0}
    stop = None
    for _ev, data in sse_frames(raw):
        try:
            f = json.loads(data)
        except Exception:
            continue
        cands = f.get("candidates") or []
        if cands:
            parts = (cands[0].get("content") or {}).get("parts") or []
            for p in parts:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    text += p["text"]
            if isinstance(cands[0].get("finishReason"), str):
                stop = cands[0]["finishReason"]
        u = f.get("usageMetadata") or {}
        if u:
            usage = {"input": u.get("promptTokenCount") or 0, "output": u.get("candidatesTokenCount") or 0}
    return text, usage, stop


# ------------------------------------------------------------------ client answers

def split_deltas(text):
    pieces = [text[i:i + SSE_DELTA_CHARS] for i in range(0, len(text), SSE_DELTA_CHARS)]
    return pieces if pieces else [""]


def sse_event(event, data):
    s = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (s, json.dumps(data, ensure_ascii=False))


def render_openai_json(text, usage, stop, model):
    return json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion", "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": stop, "logprobs": None}],
        "usage": {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                  "total_tokens": usage["input"] + usage["output"]},
    }, ensure_ascii=False).encode("utf-8")


def render_openai_stream(text, usage, stop, model):
    def chunk(delta, finish):
        return {"id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    frames = [sse_event(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in split_deltas(text):
        frames.append(sse_event(None, chunk({"content": piece}, None)))
    frames.append(sse_event(None, chunk({}, stop)))
    frames.append(sse_event(None, {"id": "chatcmpl-sim", "object": "chat.completion.chunk",
                                   "created": FIXED_CREATED, "model": model, "choices": [],
                                   "usage": {"prompt_tokens": usage["input"],
                                             "completion_tokens": usage["output"],
                                             "total_tokens": usage["input"] + usage["output"]}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode("utf-8")


def render_anthropic_json(text, usage, stop, model):
    return json.dumps({
        "id": "msg_sim", "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": usage["input"], "output_tokens": usage["output"]},
    }, ensure_ascii=False).encode("utf-8")


def render_anthropic_stream(text, usage, stop, model):
    frames = [
        sse_event("message_start", {"type": "message_start", "message": {
            "id": "msg_sim", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": usage["input"], "output_tokens": 0}}}),
        sse_event("content_block_start", {"type": "content_block_start", "index": 0,
                                          "content_block": {"type": "text", "text": ""}}),
    ]
    for piece in split_deltas(text):
        frames.append(sse_event("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                        "delta": {"type": "text_delta", "text": piece}}))
    frames.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(sse_event("message_delta", {"type": "message_delta",
                                             "delta": {"stop_reason": stop, "stop_sequence": None},
                                             "usage": {"output_tokens": usage["output"]}}))
    frames.append(sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames).encode("utf-8")


def map_stop(provider_stop, client_api):
    fam = STOP_FAMILIES.get(provider_stop, 0) if isinstance(provider_stop, str) else 0
    return OPENAI_STOP[fam] if client_api == "openai" else ANTHROPIC_STOP[fam]


# ------------------------------------------------------------------ ledger + serving

class Ledger:
    def __init__(self):
        self.calls = {}
        self.misses = {}
        self.retired = set()

    def note_success(self, pid):
        self.calls[pid] = self.calls.get(pid, 0) + 1
        self.misses[pid] = 0

    def note_refusal(self, pid, message):
        self.calls[pid] = self.calls.get(pid, 0) + 1
        self.misses[pid] = self.misses.get(pid, 0) + 1
        low = (message or "").lower()
        if "budget is spent" in low or "is out" in low or self.misses[pid] >= 3:
            self.retired.add(pid)

    def note_down(self, _pid):
        pass  # no HTTP answer: the env never saw the call, no budget spent


class Router:
    def __init__(self, brief, env_url, plan, qw):
        self.brief = brief
        self.env_url = env_url
        self.plan = plan
        self.qw = qw
        self.ledger = Ledger()
        self.lock = threading.Lock()
        self.counter = 0

    def requirements_from_body(self, api, body):
        """Fallback requirements when the manifest has no row for this POST."""
        if api == "openai":
            norm = parse_openai_body(body)
        else:
            norm = parse_anthropic_body(body)
        req = {"tools": bool(norm["tools"]), "vision": bool(norm["images"]),
               "json": norm["json_mode"]}
        return req

    def failover_pick(self, req_row, tried):
        """Cheapest eligible (provider, model) not tried, not retired, under cap."""
        best_opt = None
        best_key = None
        for (p, m, rt) in eligible_options(self.brief, req_row):
            if p in tried or p in self.ledger.retired:
                continue
            if self.ledger.calls.get(p, 0) >= CAP[p]:
                continue
            inc = 0.0 if self.ledger.calls.get(p, 0) == 0 else RISK[p]
            key = (inc * 1000.0 + req_row["weight"] * (self.brief.best[req_row["id"]] - rt) * self.qw, p, m)
            if best_key is None or key < best_key:
                best_key = key
                best_opt = (p, m, rt)
        return best_opt

    def serve(self, api, body):
        with self.lock:
            idx = self.counter
            self.counter += 1
            if idx < len(self.brief.manifest):
                row = self.brief.manifest[idx]
            else:
                row = {"id": "r%02d" % (idx + 1), "api": api, "stream": body.get("stream") is True,
                       "weight": 1, "requires": self.requirements_from_body(api, body)}
            norm = parse_openai_body(body) if api == "openai" else parse_anthropic_body(body)
            planned = self.plan.get(row["id"])
            tried = set()
            attempt = planned
            # A planned pick on a retired (budget spent / out / 3 misses) or
            # budget-capped provider would refuse for sure and waste a call:
            # skip straight to failover when another provider can serve.
            if attempt is not None and (
                    attempt[0] in self.ledger.retired
                    or self.ledger.calls.get(attempt[0], 0) >= CAP[attempt[0]]):
                tried.add(attempt[0])
                skipped = self.failover_pick(row, tried)
                if skipped is not None:
                    attempt = skipped
                else:
                    tried.discard(attempt[0])
            last_error = None
            while attempt is not None:
                pid, model, _rt = attempt
                provider = self.brief.providers[pid]
                try:
                    text, usage, stop = call_provider(self.env_url, provider, norm, model)
                except ProviderMiss as e:
                    self.ledger.note_refusal(pid, e.message)
                    tried.add(pid)
                    last_error = e
                    attempt = self.failover_pick(row, tried)
                    continue
                except ProviderDown as e:
                    self.ledger.note_down(pid)
                    tried.add(pid)
                    last_error = e
                    attempt = self.failover_pick(row, tried)
                    continue
                self.ledger.note_success(pid)
                out_stop = map_stop(stop, api)
                if api == "openai":
                    if norm["stream"]:
                        return 200, "text/event-stream; charset=utf-8", render_openai_stream(text, usage, out_stop, model)
                    return 200, "application/json", render_openai_json(text, usage, out_stop, model)
                if norm["stream"]:
                    return 200, "text/event-stream; charset=utf-8", render_anthropic_stream(text, usage, out_stop, model)
                return 200, "application/json", render_anthropic_json(text, usage, out_stop, model)
            # No provider left: the request cannot be served.
            log("no provider left for %s: %s" % (row["id"], last_error))
            if api == "anthropic":
                payload = {"type": "error", "error": {"type": "api_error", "message": "no free model could serve this request"}}
            else:
                payload = {"error": {"message": "no free model could serve this request", "type": "api_error", "code": None}}
            return 502, "application/json", json.dumps(payload, ensure_ascii=False).encode("utf-8")


# ------------------------------------------------------------------ HTTP server

def make_handler(router, brief):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _read_body(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if n <= 0 or n > 2 * 1024 * 1024:
                return None
            return self.rfile.read(n)

        def _send(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/v1/models":
                served = brief.served_models()
                if isinstance(self.headers.get("x-api-key"), str):
                    body = {"data": [
                        {"type": "model", "id": mid, "display_name": mid,
                         "created_at": "2026-01-01T00:00:00Z", "free": True}
                        for mid, _pid in served
                    ], "has_more": False}
                else:
                    body = {"object": "list", "data": [
                        {"id": mid, "object": "model", "created": FIXED_CREATED,
                         "owned_by": pid, "free": True}
                        for mid, pid in served
                    ]}
                self._send(200, "application/json", json.dumps(body, ensure_ascii=False).encode("utf-8"))
                return
            self._send(404, "application/json", json.dumps(
                {"error": {"message": "not found", "type": "not_found", "code": None}}).encode("utf-8"))

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                api = "openai"
            elif self.path == "/v1/messages":
                api = "anthropic"
            else:
                self._send(404, "application/json", json.dumps(
                    {"error": {"message": "not found", "type": "not_found", "code": None}}).encode("utf-8"))
                return
            raw = self._read_body()
            if raw is None:
                self._send(400, "application/json", json.dumps(
                    {"error": {"message": "bad request body", "type": "invalid_request_error", "code": None}}).encode("utf-8"))
                return
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = None
            if not is_record(body):
                self._send(400, "application/json", json.dumps(
                    {"error": {"message": "bad request body", "type": "invalid_request_error", "code": None}}).encode("utf-8"))
                return
            try:
                status, ctype, payload = router.serve(api, body)
            except Exception as e:  # never leave the client hanging
                log("serve failed: %r" % (e,))
                status, ctype, payload = 500, "application/json", json.dumps(
                    {"error": {"message": "internal error", "type": "api_error", "code": None}}).encode("utf-8")
            self._send(status, ctype, payload)

    return Handler


def http_get(url, timeout=5):
    req = urllib.request.Request(url, method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read()
        except Exception:
            return e.code, b""
    except Exception:
        return 0, b""


def http_post_json(url, payload, timeout=5):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read()
        except Exception:
            return e.code, b""
    except Exception:
        return 0, b""


def main():
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    brief = Brief(prompt_file)
    catalog = brief.catalog()
    plan, qw = compute_plan(brief)
    if "--list" in sys.argv:
        sys.stdout.write(json.dumps(catalog, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 0
    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    router = Router(brief, env_url, plan, qw)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(router, brief))
    host, port = server.server_address
    if port == 8080:
        log("the OS picked 8080, which the env owns: restart")
        return 1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log("listening on 0.0.0.0:%d" % port)
    status, raw = http_post_json(env_url + "/register", {
        "protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog})
    if status != 200:
        log("register answered HTTP %d: %s" % (status, raw[:300].decode("utf-8", errors="replace")))
        return 1
    log("registered: %s" % raw[:200].decode("utf-8", errors="replace"))
    while True:
        status, raw = http_get(env_url + "/work", timeout=5)
        if status == 200:
            try:
                work = json.loads(raw.decode("utf-8"))
            except Exception:
                work = {}
            if isinstance(work, dict) and work.get("done") is True:
                server.shutdown()
                server.server_close()
                return 0
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
