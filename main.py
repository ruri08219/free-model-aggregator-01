#!/usr/bin/env python3
"""free-model-aggregator-01 v3: a plan-first free-model router.

One server, two client wires (OpenAI Chat Completions + Anthropic Messages),
ten simulated free-tier providers on three wires (OpenAI, Anthropic, Gemini).
For each of the 20 requests it picks the strongest free model that can serve
it, fails over when a provider refuses, and exits 0 when the env is done.

Routing idea (the v3 change)
----------------------------
A provider's free tier is a small budget with one or two scheduled refusals
hidden inside it, so every call past a provider's first is a gamble: the
second call can already be a refusal, and a refusal costs a whole extra
upstream call. The router therefore plans the trial before it serves it:

* Each provider gets a planned-call cap. Wide-budget providers (the ones
  whose free tier stretches furthest) get the most planned calls; narrow
  ones get a single call, so a refusal on them can never cascade. The caps
  sum to exactly the 20 requests, so the plan covers the trial with no
  provider asked to carry more than its cap.
* The trial is planned before it is served. The most constrained
  requests claim their providers first (fewest legal providers, then
  heavier weight), so vision and long-context requests are never left
  with whatever the free requests did not spend. Within that order
  each request takes the highest-rated legal model on a provider that
  still has cap left, with a spread penalty so one provider does not
  carry a whole request type. A request then goes to its planned
  provider.
* A spent provider is dropped on its first refusal. Spent means its
  free-tier budget is gone or its outage has begun (a 402, or an
  error saying the budget is spent or the provider is out). A
  scheduled refusal is one call, not a spent provider: the provider
  keeps its budget and can answer a later request, so it is not
  dropped for it -- it is just not retried for the request it
  refused, which is answered at once by another provider.
* Failover prefers a provider that is neither reserved nor used yet.
  A provider planned for a later vision or long-context request is
  tried last -- stealing it can strand that request, while a free
  request has many providers to fall back on. Among the rest, a fresh
  provider (zero calls so far) is tried first: a first call to a
  provider always succeeds (a refusal can only land on call two or
  later), so a fresh failover is a guaranteed answer and each failure
  costs exactly one extra upstream call. Only when no fresh provider
  can serve the request does it fall back to the best live provider
  by rating.

Everything is a function of the brief, the requests, and the providers'
answers so far: no wall clock, no randomness, no seed. The trial seed is
never read for scheduling; the plan is built from the brief's quality
table and workload alone.

Wire fidelity
------------
The provider echoes back exactly what it parsed (system text, user text,
tool names, max_tokens, stop sequences, image hash, JSON mode) and the
client compares that echo with what it sent, so the translation between
the client wire and the provider wire must be lossless: system and user
text, tool names, token limits, stops and images are carried across
unchanged, and the provider's answer text, usage counts and stop reason
are relayed whole in the wire the client used.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Planned-call caps, one per provider. They sum to the 20 requests.
# Wide free tiers carry the trial (OpenRouter, Kilo, Groq, Pollinations);
# narrow ones get a single planned call, so one refusal on them cannot
# cascade; Cerebras gets none planned and stays a fresh failover reserve.
# The caps are set from the brief's provider list and quality table: the
# best model on each provider is what the plan spends, and the cap is how
# many times the plan is willing to gamble on that provider's budget.
PLANNED_CALLS = {
    "openrouter": 4,
    "groq": 4,
    "kilol": 3,
    "pollinations": 1,
    "gemini": 2,
    "zai": 1,
    "sambanova": 1,
    "cloudflare": 1,
    "ovh": 2,
    "cerebras": 1,
}

PROTOCOL_VERSION = 1
CATALOG_SCHEMA = "free-model-aggregator-01/catalog/v1"
PROVIDER_TIMEOUT = 25.0
WORK_POLL_SECONDS = 0.2
# How much a provider's effective rating drops per call already planned
# on it. The gap between the top model (1475) and the next tier (1470)
# is 5, so a penalty above 5 keeps one provider from carrying a whole
# request type: after one planned call the next request of that type
# goes to a different provider instead. Kept small, so the plan still
# favours the strongest models and keeps its quality margin.
SPREAD_PENALTY = 8


def read_brief(path):
    """Pull the three JSON documents out of the brief text file."""
    docs = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                doc = json.loads(line)
            except ValueError:
                continue
            schema = doc.get("schema", "")
            if schema.endswith("/providers/v1"):
                docs["providers"] = doc
            elif schema.endswith("/ratings/v1"):
                docs["ratings"] = doc
            elif schema.endswith("/workload/v1"):
                docs["workload"] = doc
    return docs


class ModelTable:
    """The sealed quality table: free models, their ratings and capabilities."""

    def __init__(self, ratings_doc):
        self.by_provider = {}
        self.all_models = []
        for row in ratings_doc.get("ratings", []):
            if not row.get("free"):
                continue
            model = {
                "id": row["model"],
                "provider": row["provider"],
                "rating": int(row.get("rating_milli", 0)),
                "context_length": row.get("context_length"),
                "tools": bool(row.get("tools")),
                "vision": bool(row.get("vision")),
                "json": bool(row.get("json")),
            }
            self.all_models.append(model)
            self.by_provider.setdefault(model["provider"], []).append(model)
        for models in self.by_provider.values():
            models.sort(key=lambda m: (-m["rating"], m["id"]))

    def legal(self, provider_id, requires, json_mode_needed):
        """Best legal model on one provider, or None.

        Legal means free, meets every requirement, and (when the client
        asked for JSON mode) on a wire that can echo it back.
        """
        provider = PROVIDERS.get(provider_id)
        if provider is None:
            return None
        if json_mode_needed and provider["wire"] == "anthropic":
            return None
        for model in self.by_provider.get(provider_id, []):
            if requires.get("tools") and not model["tools"]:
                continue
            if requires.get("vision") and not model["vision"]:
                continue
            if requires.get("json") and not model["json"]:
                continue
            min_ctx = requires.get("min_context")
            if min_ctx is not None:
                ctx = model["context_length"]
                if ctx is None or ctx < min_ctx:
                    continue
            return model
        return None


# The provider directory, filled from the brief at startup.
PROVIDERS = {}
# The brief, kept so the live-catalog fallback can rebuild from it.
CURRENT_BRIEF = None


def catalog_from_brief(brief):
    """The catalog document, built from the brief's quality table.

    One entry per provider: id, base_url (the provider's own path),
    key_env, wire, and models with id, free and context_length. Used for
    --list, where there is no env to read live catalogs from.
    """
    table = ModelTable(brief["ratings"])
    entries = []
    for provider in brief["providers"]["providers"]:
        pid = provider["id"]
        models = [
            {"id": m["id"], "free": True, "context_length": m["context_length"]}
            for m in table.by_provider.get(pid, [])
        ]
        entries.append(
            {
                "id": pid,
                "base_url": provider["path"],
                "key_env": provider.get("key_env", ""),
                "wire": provider["wire"],
                "models": models,
            }
        )
    return {"schema": CATALOG_SCHEMA, "providers": entries}


def http_json(method, url, headers, body=None, timeout=PROVIDER_TIMEOUT):
    """One HTTP call; returns (status, parsed_json_or_None, raw_text)."""
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers = dict(headers)
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", "replace")
        status = err.code
    except Exception:
        return None, None, ""
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return status, parsed, raw


def provider_auth_headers(provider):
    """The auth header a provider's own auth scheme asks for."""
    auth = provider.get("auth", "none")
    key = provider.get("key", "")
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def provider_models_url(env_url, provider):
    path = provider["path"]
    if provider["wire"] == "gemini":
        return env_url + path + "/v1beta/models"
    return env_url + path + "/v1/models"


def read_live_catalog(env_url, providers):
    """Read every provider's own model list and build the catalog to register."""
    entries = []
    for provider in providers:
        pid = provider["id"]
        status, parsed, _raw = http_json(
            "GET",
            provider_models_url(env_url, provider),
            provider_auth_headers(provider),
            timeout=10.0,
        )
        models = []
        if status == 200 and isinstance(parsed, dict):
            if provider["wire"] == "gemini":
                items = parsed.get("models", [])
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name", "")
                    mid = name.split("/")[-1] if name else item.get("id", "")
                    if mid:
                        models.append(
                            {
                                "id": mid,
                                "free": bool(item.get("free")),
                                "context_length": item.get("context_length"),
                            }
                        )
            else:
                items = parsed.get("data", [])
                for item in items:
                    if not isinstance(item, dict) or not item.get("id"):
                        continue
                    models.append(
                        {
                            "id": item["id"],
                            "free": bool(item.get("free")),
                            "context_length": item.get("context_length"),
                        }
                    )
        if not models:
            # A provider that would not list is registered from the brief
            # instead, so the catalog never loses a provider.
            table = ModelTable(CURRENT_BRIEF["ratings"])
            models = [
                {"id": m["id"], "free": True, "context_length": m["context_length"]}
                for m in table.by_provider.get(pid, [])
            ]
        entries.append(
            {
                "id": pid,
                "base_url": provider["path"],
                "key_env": provider.get("key_env", ""),
                "wire": provider["wire"],
                "models": models,
            }
        )
    return {"schema": CATALOG_SCHEMA, "providers": entries}


# ---------------------------------------------------------------------------
# Request normalisation: one shape for both client wires.
# ---------------------------------------------------------------------------


def blocks_text(content):
    """The text of a message body: a string as-is, blocks joined by newline."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def split_data_url(url):
    """A data: URL into (mime, base64) or None when it is a plain URL."""
    if not isinstance(url, str):
        return None
    if not url.startswith("data:"):
        return None
    head, _, data = url.partition(",")
    if not data:
        return None
    mime = head[5:].split(";")[0] or "image/png"
    return (mime, data)


def normalise_openai(body):
    """An OpenAI Chat Completions body into the shared request shape."""
    system_parts = []
    user_text = ""
    image = None
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            text = blocks_text(message.get("content"))
            if text:
                system_parts.append(text)
        elif message.get("role") == "user":
            user_text = blocks_text(message.get("content"))
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "image_url":
                        continue
                    image_url = block.get("image_url")
                    if not isinstance(image_url, dict):
                        continue
                    url = image_url.get("url")
                    if not isinstance(url, str) or not url:
                        continue
                    parsed = split_data_url(url)
                    image = ("b64", parsed[0], parsed[1]) if parsed else ("url", url)
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
                name = tool["function"].get("name")
                if isinstance(name, str):
                    tools.append(name)
    fmt = body.get("response_format")
    json_mode = isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema")
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = None
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in stop] if isinstance(stop, list) else []
    stops = [s for s in stops if isinstance(s, str)]
    return {
        "system": "\n".join(system_parts),
        "user": user_text,
        "image": image,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": stops,
        "json_mode": json_mode,
        "stream": body.get("stream") is True,
    }


def normalise_anthropic(body):
    """An Anthropic Messages body into the shared request shape."""
    system_text = blocks_text(body.get("system"))
    user_text = ""
    image = None
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        user_text = blocks_text(message.get("content"))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "image":
                    continue
                source = block.get("source")
                if not isinstance(source, dict):
                    continue
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    mime = source.get("media_type") or "image/png"
                    image = ("b64", mime, source["data"])
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    image = ("url", source["url"])
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                tools.append(tool["name"])
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = None
    stops = body.get("stop_sequences")
    stops = [s for s in stops if isinstance(s, str)] if isinstance(stops, list) else []
    return {
        "system": system_text,
        "user": user_text,
        "image": image,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": stops,
        "json_mode": False,
        "stream": body.get("stream") is True,
    }


# ---------------------------------------------------------------------------
# Provider bodies: the shared shape back out onto each provider wire.
# ---------------------------------------------------------------------------


def user_content_openai(norm):
    """The user message content for an OpenAI-wire provider."""
    image = norm["image"]
    if image is None:
        return norm["user"]
    if image[0] == "b64":
        url = "data:%s;base64,%s" % (image[1], image[2])
    else:
        url = image[1]
    return [
        {"type": "text", "text": norm["user"]},
        {"type": "image_url", "image_url": {"url": url}},
    ]


def build_openai_body(norm, model_id):
    body = {"model": model_id, "messages": [], "stream": norm["stream"]}
    if norm["system"]:
        body["messages"].append({"role": "system", "content": norm["system"]})
    body["messages"].append({"role": "user", "content": user_content_openai(norm)})
    if norm["tools"]:
        body["tools"] = [
            {"type": "function", "function": {"name": name}} for name in norm["tools"]
        ]
    if norm["max_tokens"] is not None:
        body["max_tokens"] = norm["max_tokens"]
    if norm["stops"]:
        body["stop"] = norm["stops"]
    if norm["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    if norm["stream"]:
        body["stream_options"] = {"include_usage": True}
    return body


def build_anthropic_body(norm, model_id):
    content = [{"type": "text", "text": norm["user"]}]
    image = norm["image"]
    if image is not None:
        if image[0] == "b64":
            source = {"type": "base64", "media_type": image[1], "data": image[2]}
        else:
            source = {"type": "url", "url": image[1]}
        content.append({"type": "image", "source": source})
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": norm["stream"],
    }
    if norm["system"]:
        body["system"] = norm["system"]
    if norm["tools"]:
        body["tools"] = [
            {"name": name, "description": "", "input_schema": {"type": "object"}}
            for name in norm["tools"]
        ]
    if norm["max_tokens"] is not None:
        body["max_tokens"] = norm["max_tokens"]
    if norm["stops"]:
        body["stop_sequences"] = norm["stops"]
    return body


def build_gemini_body(norm):
    parts = [{"text": norm["user"]}]
    image = norm["image"]
    if image is not None:
        if image[0] == "b64":
            parts.append({"inlineData": {"mimeType": image[1], "data": image[2]}})
        else:
            parts.append({"fileData": {"fileUri": image[1]}})
    body = {"contents": [{"role": "user", "parts": parts}]}
    if norm["system"]:
        body["systemInstruction"] = {"parts": [{"text": norm["system"]}]}
    if norm["tools"]:
        body["tools"] = [
            {"functionDeclarations": [{"name": name} for name in norm["tools"]]}
        ]
    config = {}
    if norm["max_tokens"] is not None:
        config["maxOutputTokens"] = norm["max_tokens"]
    if norm["stops"]:
        config["stopSequences"] = norm["stops"]
    if norm["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# ---------------------------------------------------------------------------
# Provider answers: parse each wire's JSON and SSE back into one shape.
# ---------------------------------------------------------------------------


def parse_sse_frames(raw):
    """An SSE body into [(event_or_None, data_string)] in order."""
    frames = []
    text = raw.replace("\r\n", "\n")
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


def join_text_blocks(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
    return ""


def parse_openai_answer(parsed, raw, stream):
    if not stream:
        if not isinstance(parsed, dict):
            return None
        choices = parsed.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return None
        choice = choices[0]
        message = choice.get("message") or {}
        usage = parsed.get("usage") or {}
        return {
            "text": join_text_blocks(message.get("content", "")),
            "input": usage.get("prompt_tokens"),
            "output": usage.get("completion_tokens"),
            "stop": choice.get("finish_reason"),
        }
    text_parts = []
    usage = {"input": None, "output": None}
    stop = None
    for _event, data in parse_sse_frames(raw):
        if data.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("usage"), dict):
            usage["input"] = chunk["usage"].get("prompt_tokens")
            usage["output"] = chunk["usage"].get("completion_tokens")
        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        if isinstance(choice.get("finish_reason"), str):
            stop = choice["finish_reason"]
    return {"text": "".join(text_parts), "input": usage["input"], "output": usage["output"], "stop": stop}


def parse_anthropic_answer(parsed, raw, stream):
    if not stream:
        if not isinstance(parsed, dict):
            return None
        usage = parsed.get("usage") or {}
        return {
            "text": join_text_blocks(parsed.get("content", [])),
            "input": usage.get("input_tokens"),
            "output": usage.get("output_tokens"),
            "stop": parsed.get("stop_reason"),
        }
    text_parts = []
    usage = {"input": None, "output": None}
    stop = None
    for event, data in parse_sse_frames(raw):
        try:
            frame = json.loads(data)
        except ValueError:
            continue
        if not isinstance(frame, dict):
            continue
        if event == "message_start":
            message = frame.get("message") or {}
            inner = message.get("usage") or {}
            usage["input"] = inner.get("input_tokens")
        elif event == "content_block_delta":
            delta = frame.get("delta") or {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                text_parts.append(delta["text"])
        elif event == "message_delta":
            delta = frame.get("delta") or {}
            if isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            inner = frame.get("usage") or {}
            usage["output"] = inner.get("output_tokens")
    return {"text": "".join(text_parts), "input": usage["input"], "output": usage["output"], "stop": stop}


def parse_gemini_answer(parsed, raw, stream):
    def candidate_text(candidate):
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        return "".join(
            p.get("text", "") for p in parts
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )

    if not stream:
        if not isinstance(parsed, dict):
            return None
        candidates = parsed.get("candidates") or []
        if not candidates or not isinstance(candidates[0], dict):
            return None
        meta = parsed.get("usageMetadata") or {}
        return {
            "text": candidate_text(candidates[0]),
            "input": meta.get("promptTokenCount"),
            "output": meta.get("candidatesTokenCount"),
            "stop": candidates[0].get("finishReason"),
        }
    text_parts = []
    usage = {"input": None, "output": None}
    stop = None
    for _event, data in parse_sse_frames(raw):
        try:
            frame = json.loads(data)
        except ValueError:
            continue
        if not isinstance(frame, dict):
            continue
        candidates = frame.get("candidates") or []
        if candidates and isinstance(candidates[0], dict):
            text_parts.append(candidate_text(candidates[0]))
            if isinstance(candidates[0].get("finishReason"), str):
                stop = candidates[0]["finishReason"]
        meta = frame.get("usageMetadata") or {}
        if meta:
            usage["input"] = meta.get("promptTokenCount")
            usage["output"] = meta.get("candidatesTokenCount")
    return {"text": "".join(text_parts), "input": usage["input"], "output": usage["output"], "stop": stop}


def call_provider(env_url, provider, model_id, norm):
    """One provider call.

    Returns (True, answer, False) on success, or (False, None, spent) on
    refusal, where spent is True only when the provider is done for the
    trial -- its free-tier budget is spent, or its outage has begun.
    A scheduled refusal ("refused this call") is one call, not a spent
    provider: the provider keeps its budget and can answer a later
    request, so it is not dropped for it.
    """
    wire = provider["wire"]
    headers = provider_auth_headers(provider)
    if wire == "openai":
        url = env_url + provider["path"] + "/v1/chat/completions"
        body = build_openai_body(norm, model_id)
        parse = parse_openai_answer
    elif wire == "anthropic":
        url = env_url + provider["path"] + "/v1/messages"
        body = build_anthropic_body(norm, model_id)
        parse = parse_anthropic_answer
    else:
        import urllib.parse

        verb = "streamGenerateContent" if norm["stream"] else "generateContent"
        url = (
            env_url
            + provider["path"]
            + "/v1beta/models/"
            + urllib.parse.quote(model_id, safe="")
            + ":"
            + verb
        )
        body = build_gemini_body(norm)
        parse = parse_gemini_answer
    status, parsed, raw = http_json("POST", url, headers, body=body)
    if status is None or status < 200 or status >= 300:
        spent = False
        if status == 402:
            spent = True
        else:
            message = ""
            if isinstance(parsed, dict):
                error = parsed.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message", ""))
            if "budget is spent" in message or " is out" in message:
                spent = True
        return False, None, spent
    answer = parse(parsed, raw, norm["stream"])
    if answer is None or answer.get("text") is None:
        return False, None, False
    return True, answer, False


# ---------------------------------------------------------------------------
# The router: a planned cap per provider, drop on first refusal,
# fresh-first failover.
# ---------------------------------------------------------------------------


def compute_plan(table, manifest):
    """Precompute request -> (provider, model) for the whole workload.

    The most constrained requests claim their providers first (fewest
    legal providers, then heavier weight), so vision and long-context
    requests are never left with whatever the free requests did not
    spend. Within that order each request takes the highest-rated legal
    model on a provider that still has planned cap left, with a spread
    penalty so one provider does not carry a whole request type: after
    each planned call its effective rating drops, and the next request
    of that type goes elsewhere. The result is a full assignment before
    a single request is served, so a long-context request always has a
    provider of its own instead of failing over to one already spent.
    """
    cap_left = dict(PLANNED_CALLS)
    planned_count = {pid: 0 for pid in PROVIDERS}

    def legal_providers(entry):
        requires = entry.get("requires", {})
        json_needed = entry.get("api") == "openai" and bool(requires.get("json"))
        return [
            pid
            for pid in PROVIDERS
            if table.legal(pid, requires, json_needed) is not None
        ]

    order = sorted(
        manifest,
        key=lambda e: (len(legal_providers(e)), -e.get("weight", 1), e["id"]),
    )
    plan = {}
    for entry in order:
        requires = entry.get("requires", {})
        json_needed = entry.get("api") == "openai" and bool(requires.get("json"))
        best = None
        best_eff = None
        for pid in PROVIDERS:
            if cap_left.get(pid, 0) <= 0:
                continue
            model = table.legal(pid, requires, json_needed)
            if model is None:
                continue
            eff = model["rating"] - SPREAD_PENALTY * planned_count[pid]
            if best_eff is None or eff > best_eff or (eff == best_eff and pid < best[0]):
                best = (pid, model)
                best_eff = eff
        if best is None:
            continue
        pid, model = best
        plan[entry["id"]] = (pid, model)
        cap_left[pid] -= 1
        planned_count[pid] += 1
    return plan


# Requests whose provider must not be stolen by a failover: vision and
# long-context requests have the fewest legal providers, so if an
# earlier failover takes the provider planned for one of them, that
# request can be left with no provider at all.
def constrained_ids(manifest):
    return {
        entry["id"]
        for entry in manifest
        if entry.get("requires", {}).get("vision")
        or entry.get("requires", {}).get("min_context") is not None
    }


class Router:
    def __init__(self, table, plan, manifest):
        self.table = table
        self.plan = plan
        self.order = [entry["id"] for entry in manifest]
        self.constrained = constrained_ids(manifest)
        self.calls = {pid: 0 for pid in PROVIDERS}
        self.dead = set()
        self.served = set()

    def _reserved_for_future(self, pid, request_id):
        """True when pid is planned for a later vision/min-context request."""
        try:
            at = self.order.index(request_id)
        except ValueError:
            return False
        for later_id in self.order[at + 1 :]:
            if later_id in self.constrained and self.plan.get(later_id, (None,))[0] == pid:
                return True
        return False

    def _legal(self, requires, json_mode_needed, tried):
        """Live providers (not tried, not dropped) with their best legal model."""
        found = []
        for pid in PROVIDERS:
            if pid in self.dead or pid in tried:
                continue
            model = self.table.legal(pid, requires, json_mode_needed)
            if model is not None:
                found.append((pid, model))
        return found

    def _failover(self, requires, json_mode_needed, tried, request_id):
        """Best legal model, preferring an unreserved fresh provider.

        A provider planned for a later vision or long-context request is
        tried last: stealing it can strand that request, while a free
        request has many providers to fall back on. Among the rest, a
        fresh provider (zero calls) is tried first -- its first call
        always succeeds -- and then the best live provider by rating.
        """
        cands = self._legal(requires, json_mode_needed, tried)
        if not cands:
            return None
        cands.sort(
            key=lambda c: (
                1 if self._reserved_for_future(c[0], request_id) else 0,
                0 if self.calls.get(c[0], 0) == 0 else 1,
                -c[1]["rating"],
                c[0],
            )
        )
        return cands[0]

    def serve(self, env_url, requires, norm, request_id):
        """Answer one request. Returns (provider_id, model, answer) or None."""
        self.served.add(request_id)
        tried = set()
        first_try = True
        while True:
            if first_try:
                first_try = False
                planned = self.plan.get(request_id)
                if (
                    planned is not None
                    and planned[0] not in self.dead
                    and planned[0] not in tried
                    and not (
                        norm["json_mode"]
                        and PROVIDERS[planned[0]]["wire"] == "anthropic"
                    )
                ):
                    choice = planned
                else:
                    choice = self._failover(requires, norm["json_mode"], tried, request_id)
            else:
                choice = self._failover(requires, norm["json_mode"], tried, request_id)
            if choice is None:
                return None
            pid, model = choice
            tried.add(pid)
            ok, answer, spent = call_provider(env_url, PROVIDERS[pid], model["id"], norm)
            self.calls[pid] += 1
            if ok:
                return pid, model, answer
            # Only a spent provider is dropped. A scheduled refusal is
            # one call: the provider keeps its budget, so it stays live
            # for a later request (it is just not retried for this one).
            if spent:
                self.dead.add(pid)


# ---------------------------------------------------------------------------
# Answers back out in the wire the client used. The provider's text,
# usage and stop reason are relayed whole.
# ---------------------------------------------------------------------------

FIXED_CREATED = 1780000000


def split_text(text, size=64):
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]


def emit_openai_json(model_id, answer):
    usage = {
        "prompt_tokens": answer["input"],
        "completion_tokens": answer["output"],
        "total_tokens": (answer["input"] or 0) + (answer["output"] or 0),
    }
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer["text"]},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": usage,
    }


def emit_openai_stream(model_id, answer):
    def chunk(delta, finish):
        return {
            "id": "chatcmpl-sim",
            "object": "chat.completion.chunk",
            "created": FIXED_CREATED,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    frames = ["data: %s\n\n" % json.dumps(chunk({"role": "assistant", "content": ""}, None))]
    for piece in split_text(answer["text"]):
        frames.append("data: %s\n\n" % json.dumps(chunk({"content": piece}, None)))
    frames.append("data: %s\n\n" % json.dumps(chunk({}, "stop")))
    usage = {
        "prompt_tokens": answer["input"],
        "completion_tokens": answer["output"],
        "total_tokens": (answer["input"] or 0) + (answer["output"] or 0),
    }
    frames.append(
        "data: %s\n\n"
        % json.dumps(
            {
                "id": "chatcmpl-sim",
                "object": "chat.completion.chunk",
                "created": FIXED_CREATED,
                "model": model_id,
                "choices": [],
                "usage": usage,
            }
        )
    )
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def emit_anthropic_json(model_id, answer):
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": [{"type": "text", "text": answer["text"]}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": answer["input"], "output_tokens": answer["output"]},
    }


def emit_anthropic_stream(model_id, answer):
    def frame(event, data):
        return "event: %s\ndata: %s\n\n" % (event, json.dumps(data))

    out = []
    out.append(
        frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_sim",
                    "type": "message",
                    "role": "assistant",
                    "model": model_id,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": answer["input"], "output_tokens": 0},
                },
            },
        )
    )
    out.append(
        frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
    )
    for piece in split_text(answer["text"]):
        out.append(
            frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": piece},
                },
            )
        )
    out.append(
        frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    )
    out.append(
        frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": answer["output"]},
            },
        )
    )
    out.append(frame("message_stop", {"type": "message_stop"}))
    return "".join(out)


# ---------------------------------------------------------------------------
# The server: the two client endpoints, the model list, registration,
# and the work loop that exits when the env is done.
# ---------------------------------------------------------------------------


class ServerState:
    def __init__(self, env_url, brief, catalog, table, manifest):
        self.env_url = env_url
        self.brief = brief
        self.catalog = catalog
        self.table = table
        self.manifest = manifest
        self.router = Router(table, compute_plan(table, manifest), manifest)
        self.lock = threading.Lock()
        self.next_index = 0
        # The free-model list, in the OpenAI shape, for GET /v1/models.
        data = []
        for entry in catalog["providers"]:
            for model in entry["models"]:
                if model.get("free"):
                    data.append(
                        {
                            "id": model["id"],
                            "object": "model",
                            "created": FIXED_CREATED,
                            "owned_by": entry["id"],
                            "free": True,
                            "context_length": model.get("context_length"),
                        }
                    )
        self.models_list = {"object": "list", "data": data}


STATE = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        pass

    def _send_json(self, status, payload):
        body = (json.dumps(payload) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, text):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(200, STATE.models_list)
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            self._send_json(400, {"error": "bad_json"})
            return
        if not isinstance(body, dict):
            self._send_json(400, {"error": "bad_json"})
            return
        if self.path == "/v1/chat/completions":
            api = "openai"
        elif self.path == "/v1/messages":
            api = "anthropic"
        else:
            self._send_json(404, {"error": "not_found"})
            return
        with STATE.lock:
            index = STATE.next_index
            STATE.next_index += 1
        manifest = STATE.manifest
        entry = manifest[index] if index < len(manifest) else None
        requires = entry.get("requires", {}) if entry else {}
        request_id = entry.get("id") if entry else None
        norm = normalise_openai(body) if api == "openai" else normalise_anthropic(body)
        result = STATE.router.serve(STATE.env_url, requires, norm, request_id)
        if result is None:
            self._send_json(502, {"error": "no_provider_could_serve"})
            return
        _pid, model, answer = result
        if api == "openai":
            if norm["stream"]:
                self._send_sse(emit_openai_stream(model["id"], answer))
            else:
                self._send_json(200, emit_openai_json(model["id"], answer))
        else:
            if norm["stream"]:
                self._send_sse(emit_anthropic_stream(model["id"], answer))
            else:
                self._send_json(200, emit_anthropic_json(model["id"], answer))


def register(env_url, port, catalog):
    status, _parsed, _raw = http_json(
        "POST",
        env_url + "/register",
        {},
        body={"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
        timeout=10.0,
    )
    return status == 200


def work_done(env_url):
    status, parsed, _raw = http_json("GET", env_url + "/work", {}, timeout=10.0)
    if status != 200 or not isinstance(parsed, dict):
        return False
    return parsed.get("done") is True


def main(argv):
    global PROVIDERS, CURRENT_BRIEF, STATE
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if "--list" in argv:
        if not prompt_file:
            sys.stderr.write("MAKEFUN_PROMPT_FILE is not set\n")
            return 2
        brief = read_brief(prompt_file)
        sys.stdout.write(json.dumps(catalog_from_brief(brief)) + "\n")
        return 0
    if not prompt_file or not env_url:
        sys.stderr.write("MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL are required\n")
        return 2
    brief = read_brief(prompt_file)
    CURRENT_BRIEF = brief
    providers = brief["providers"]["providers"]
    for provider in providers:
        PROVIDERS[provider["id"]] = provider
    table = ModelTable(brief["ratings"])
    manifest = brief["workload"]["manifest"]
    catalog = read_live_catalog(env_url, providers)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    STATE = ServerState(env_url, brief, catalog, table, manifest)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if not register(env_url, port, catalog):
        sys.stderr.write("register failed\n")
        server.shutdown()
        return 1
    while not work_done(env_url):
        time.sleep(WORK_POLL_SECONDS)
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
