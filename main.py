#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

Serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
server-sent events) over the ten simulated free-tier providers the trial's brief
names, routing each request to the best free model that meets its requirements
and failing over when a provider refuses.

Design (all deterministic: no wall clock, no unseeded randomness):
  * The brief (MAKEFUN_PROMPT_FILE) carries the provider directory, the quality
    table and the workload manifest. The server probes each provider's catalog
    route once, builds one catalog, prints it for --list and registers it.
  * Routing ranks every free model meeting a request's requirements by the
    quality table's rating_milli, breaking ties toward the provider called
    least so far (spreading load across equal-rated models). A JSON-mode
    request never goes to an anthropic-wire provider: that wire cannot carry
    response_format, so its echo would report json_mode absent.
  * Failover reads the provider's error message, not just its status. The env
    refuses with three distinct messages: "budget is spent" and "is out" mean
    the provider serves no more of this trial, so it is never called again;
    "refused this call" is one scheduled refusal, so the provider stays in
    the rotation. Three consecutive failures also retire a provider (the
    generator draws at most two scheduled refusals per provider, so a third
    consecutive failure can only be exhaustion or an outage). The first
    successful call ends the request: exactly one provider call answers it.
  * Every wire is translated losslessly for the fields the client checks:
    the system prompt, the user text, the image bytes, the tool names,
    max_tokens, the stop sequences and JSON mode all reach the provider
    unchanged, and the provider's answer text, usage counts and stop reason
    come back in the wire the client used. OpenAI streaming requests set
    stream_options.include_usage so the provider's stream carries usage.
  * The server polls GET /work and exits 0 the moment the env says done.

Python 3.13, standard library only.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
PROVIDER_TIMEOUT_S = 10.0
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_ATTEMPTS_PER_REQUEST = 12

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


def str_of(v):
    return v if isinstance(v, str) else ""


def num_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


# ---------------------------------------------------------------------------
# The brief: three JSON documents found by their `schema` member.
# ---------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _provider_of(v):
    if not is_record(v):
        return None
    pid = str_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if str_of(v.get(key)) != "":
            base = str_of(v.get(key))
            break
    return {
        "id": pid,
        "name": str_of(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": str_of(v.get("key_env")),
        "key": str_of(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = str_of(v.get("model"))
    provider = str_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) else 0,
        "context_length": num_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not is_record(v):
        return None
    rid = str_of(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if is_record(v.get("requires")) else {}
    req = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    mc = requires.get("min_context")
    if isinstance(mc, (int, float)):
        req["min_context"] = mc
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": req,
    }


def read_brief(path):
    """The brief at `path`: providers, ratings, workload manifest."""
    brief = {"providers": [], "ratings": None, "workload": None, "error": None}
    if path == "":
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = _documents_in(text)
    directory = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
            break
    if directory is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    providers = []
    for raw in directory["providers"]:
        entry = _provider_of(raw)
        if entry is not None:
            providers.append(entry)
    brief["providers"] = providers
    ratings_doc = None
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
            break
    if ratings_doc is not None:
        rows = []
        for raw in ratings_doc["ratings"]:
            row = _rating_of(raw)
            if row is not None:
                rows.append(row)
        best = []
        if isinstance(ratings_doc.get("best"), list):
            for raw in ratings_doc["best"]:
                if is_record(raw) and str_of(raw.get("id")) != "":
                    best.append({"id": str_of(raw.get("id")),
                                 "rating_milli": num_or_none(raw.get("rating_milli")) or 0})
        brief["ratings"] = {"ratings": rows, "best": best}
    workload_doc = None
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
            break
    if workload_doc is not None:
        manifest = []
        for raw in workload_doc["manifest"]:
            entry = _workload_request_of(raw)
            if entry is not None:
                manifest.append(entry)
        brief["workload"] = {"manifest": manifest}
    return brief


def rating_of_model(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def join_base(env_url, base):
    """A base path joined to the env's URL; an absolute base is used as-is."""
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ---------------------------------------------------------------------------
# The catalog probe: each provider's catalog route, in its own wire.
# ---------------------------------------------------------------------------

def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v.get("pricing")) else None
    if pricing is None:
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        model_id = str_of(v.get("name")).replace("models/", "", 1)
    else:
        model_id = str_of(v.get("id"))
    if model_id == "":
        return None
    context = num_or_none(v.get("context_length"))
    if context is None:
        context = num_or_none(v.get("context_window"))
    if context is None:
        context = num_or_none(v.get("inputTokenLimit"))
    return {
        "id": model_id,
        "display_name": str_of(v.get("display_name")) or str_of(v.get("displayName")) or model_id,
        "free": bool_or_none(v.get("free")) if v.get("free") is not None else _zero_priced(v),
        "context_length": context,
        "tools": bool_or_none(v.get("tools")),
        "vision": bool_or_none(v.get("vision")),
        "json": bool_or_none(v.get("json")),
    }


def _models_in(wire, body):
    if not is_record(body):
        return []
    listing = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return []
    out = []
    for raw in listing:
        model = _model_in(wire, raw)
        if model is not None:
            out.append(model)
    return out


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """One provider's catalog route. Returns (ok, status, [models], error)."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return (False, resp.status, [], "GET %s answered HTTP %d" % (url, resp.status))
            body = json.loads(resp.read().decode("utf-8"))
        return (True, 200, _models_in(provider["wire"], body), None)
    except urllib.error.HTTPError as exc:
        return (False, exc.code, [], "GET %s answered HTTP %d" % (url, exc.code))
    except Exception as exc:
        return (False, 0, [], "GET %s failed: %s" % (url, exc))


def model_free(model, row):
    """Whether a model is free: the quality table's row where it has one."""
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    """What a model can do: the quality table's row first, the listing second."""
    return {
        "context_length": (row["context_length"] if row is not None else None)
                            if row is not None and row["context_length"] is not None
                            else model["context_length"],
        "tools": row["tools"] if row is not None else model["tools"],
        "vision": row["vision"] if row is not None else model["vision"],
        "json": row["json"] if row is not None else model["json"],
    }


def meets_requirements(requires, caps):
    """Whether a model's capabilities meet one request's requirements."""
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps["context_length"] is None or caps["context_length"] < mc):
        return False
    return True


def build_catalog(probes, ratings):
    """The catalog the probes and the quality table make, in brief order."""
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            context = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({
                "id": model["id"],
                "free": model_free(model, row),
                "context_length": context,
            })
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


# ---------------------------------------------------------------------------
# Routing: ranked candidates, and what each provider's failures taught us.
# ---------------------------------------------------------------------------

class RouterState:
    """Per-provider call counts, consecutive failures and retired providers."""

    def __init__(self):
        self.calls = {}
        self.consecutive_fails = {}
        self.dead = set()

    def record_call(self, provider_id):
        self.calls[provider_id] = self.calls.get(provider_id, 0) + 1

    def record_success(self, provider_id):
        self.consecutive_fails[provider_id] = 0

    def record_failure(self, provider_id, message):
        self.consecutive_fails[provider_id] = self.consecutive_fails.get(provider_id, 0) + 1
        text = message or ""
        if "budget is spent" in text or " is out" in text:
            # The provider serves no more of this trial (budget spent or outage).
            self.dead.add(provider_id)
        elif self.consecutive_fails[provider_id] >= 3:
            # The generator draws at most two scheduled refusals per provider,
            # so a third consecutive failure can only be exhaustion or outage.
            self.dead.add(provider_id)

    def is_dead(self, provider_id):
        return provider_id in self.dead

    def calls_made(self, provider_id):
        return self.calls.get(provider_id, 0)


def candidates_for(probes, ratings, requires, json_mode, router):
    """Free models meeting the requirements, best rating first.

    Ties break toward the provider called least so far, then the brief's
    provider order, then the model id: deterministic, and it spreads load
    across equal-rated models. A JSON-mode request never goes to an
    anthropic-wire provider: that wire cannot carry response_format.
    """
    out = []
    for index, probe in enumerate(probes):
        if not probe["ok"]:
            continue
        provider = probe["provider"]
        if router.is_dead(provider["id"]):
            continue
        if json_mode and provider["wire"] == "anthropic":
            continue
        for model in probe["models"]:
            row = rating_of_model(ratings, provider["id"], model["id"])
            if not model_free(model, row):
                continue
            if not meets_requirements(requires, capabilities_of(model, row)):
                continue
            rating = row["rating_milli"] if row is not None else 0
            out.append({
                "rating": rating,
                "provider": provider,
                "model": model,
                "probe_index": index,
            })
    out.sort(key=lambda c: (-c["rating"], router.calls_made(c["provider"]["id"]),
                            c["probe_index"], c["model"]["id"]))
    return out


def served_models(probes, ratings):
    """The free models, in brief order, each id once: what GET /v1/models lists."""
    out = []
    seen = set()
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            out.append({
                "id": model["id"],
                "provider": probe["provider"]["id"],
                "display_name": model["display_name"],
            })
    return out


# ---------------------------------------------------------------------------
# Wire translation. A request is normalized once, then built in whichever
# wire serves it; the fields the client checks (system, user text, image,
# tool names, max_tokens, stops, JSON mode) reach the provider unchanged.
# ---------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def _image_part(block):
    """An image block as {mime, data} for a data URL, or {url} for a plain URL."""
    if not is_record(block):
        return None
    if block.get("type") == "image_url" and is_record(block.get("image_url")):
        url = str_of(block["image_url"].get("url"))
        if url.startswith("data:"):
            head, _, data = url[5:].partition(",")
            mime = head.split(";")[0] or "image/png"
            return {"mime": mime, "data": data}
        if url != "":
            return {"url": url}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            return {"mime": str_of(source.get("media_type")) or "image/png",
                    "data": source["data"]}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"url": source["url"]}
    return None


def normalize_openai(body):
    """An OpenAI Chat Completions body as the normalized request."""
    norm = {"system": "", "turns": [], "tools": [], "tool_choice": None,
            "max_tokens": None, "temperature": None, "top_p": None,
            "stop": [], "json_mode": False, "stream": body.get("stream") is True}
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    system_parts = []
    for raw in messages:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            text = _text_of(raw.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        if role == "user":
            parts = []
            content = raw.get("content")
            if isinstance(content, str):
                parts.append({"kind": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"kind": "text", "text": block["text"]})
                    else:
                        image = _image_part(block)
                        if image is not None:
                            parts.append({"kind": "image", "image": image})
            norm["turns"].append({"role": "user", "parts": parts})
        elif role == "assistant":
            calls = []
            for call in raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                calls.append({"id": str_of(call.get("id")),
                              "name": str_of(fn.get("name")),
                              "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments") or {})})
            norm["turns"].append({"role": "assistant", "text": _text_of(raw.get("content")),
                                  "tool_calls": calls})
        elif role == "tool":
            norm["turns"].append({"role": "tool",
                                  "tool_call_id": str_of(raw.get("tool_call_id")),
                                  "text": _tool_result_text(raw.get("content"))})
    norm["system"] = "\n".join(system_parts)
    for raw in body.get("tools") if isinstance(body.get("tools"), list) else []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if str_of(fn.get("name")) == "":
            continue
        norm["tools"].append({"name": str_of(fn.get("name")),
                              "description": str_of(fn.get("description")),
                              "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}}})
    choice = body.get("tool_choice")
    if choice == "auto":
        norm["tool_choice"] = {"kind": "auto"}
    elif choice == "required":
        norm["tool_choice"] = {"kind": "any"}
    elif choice == "none":
        norm["tool_choice"] = {"kind": "none"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        norm["tool_choice"] = {"kind": "tool", "name": str_of(choice["function"].get("name"))}
    max_tokens = num_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = num_or_none(body.get("max_completion_tokens"))
    norm["max_tokens"] = max_tokens
    norm["temperature"] = num_or_none(body.get("temperature"))
    norm["top_p"] = num_or_none(body.get("top_p"))
    stop = body.get("stop")
    if isinstance(stop, str):
        norm["stop"] = [stop]
    elif isinstance(stop, list):
        norm["stop"] = [s for s in stop if isinstance(s, str)]
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    norm["json_mode"] = fmt is not None and fmt.get("type") in ("json_object", "json_schema")
    return norm


def normalize_anthropic(body):
    """An Anthropic Messages body as the normalized request."""
    norm = {"system": "", "turns": [], "tools": [], "tool_choice": None,
            "max_tokens": None, "temperature": None, "top_p": None,
            "stop": [], "json_mode": False, "stream": body.get("stream") is True}
    norm["system"] = _text_of(body.get("system"))
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in messages:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "user":
            parts = []
            if isinstance(content, str):
                parts.append({"kind": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"kind": "text", "text": block["text"]})
                    elif block.get("type") == "tool_result":
                        norm["turns"].append({"role": "tool",
                                              "tool_call_id": str_of(block.get("tool_use_id")),
                                              "text": _tool_result_text(block.get("content"))})
                    else:
                        image = _image_part(block)
                        if image is not None:
                            parts.append({"kind": "image", "image": image})
            if parts:
                norm["turns"].append({"role": "user", "parts": parts})
        elif role == "assistant":
            calls = []
            text_parts = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        calls.append({"id": str_of(block.get("id")),
                                      "name": str_of(block.get("name")),
                                      "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {})})
            norm["turns"].append({"role": "assistant", "text": "\n".join(text_parts),
                                  "tool_calls": calls})
    for raw in body.get("tools") if isinstance(body.get("tools"), list) else []:
        if not is_record(raw) or str_of(raw.get("name")) == "":
            continue
        norm["tools"].append({"name": str_of(raw.get("name")),
                              "description": str_of(raw.get("description")),
                              "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}}})
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            norm["tool_choice"] = {"kind": "auto"}
        elif choice.get("type") == "any":
            norm["tool_choice"] = {"kind": "any"}
        elif choice.get("type") == "tool":
            norm["tool_choice"] = {"kind": "tool", "name": str_of(choice.get("name"))}
    norm["max_tokens"] = num_or_none(body.get("max_tokens"))
    norm["temperature"] = num_or_none(body.get("temperature"))
    norm["top_p"] = num_or_none(body.get("top_p"))
    stops = body.get("stop_sequences")
    if isinstance(stops, list):
        norm["stop"] = [s for s in stops if isinstance(s, str)]
    return norm


def _openai_parts(parts):
    """Normalized user parts as OpenAI content parts."""
    out = []
    for part in parts:
        if part["kind"] == "text":
            out.append({"type": "text", "text": part["text"]})
        else:
            image = part["image"]
            if "data" in image:
                out.append({"type": "image_url",
                            "image_url": {"url": "data:%s;base64,%s" % (image["mime"], image["data"])}})
            else:
                out.append({"type": "image_url", "image_url": {"url": image["url"]}})
    return out


def to_openai(norm, model):
    """The normalized request as an OpenAI Chat Completions body."""
    messages = []
    if norm["system"] != "":
        messages.append({"role": "system", "content": norm["system"]})
    for turn in norm["turns"]:
        if turn["role"] == "user":
            parts = _openai_parts(turn["parts"])
            if len(parts) == 1 and parts[0]["type"] == "text":
                messages.append({"role": "user", "content": parts[0]["text"]})
            elif parts:
                messages.append({"role": "user", "content": parts})
        elif turn["role"] == "assistant":
            message = {"role": "assistant", "content": turn["text"] if turn["text"] != "" else None}
            if turn["tool_calls"]:
                message["tool_calls"] = [
                    {"id": call["id"], "type": "function",
                     "function": {"name": call["name"], "arguments": call["arguments"]}}
                    for call in turn["tool_calls"]]
            messages.append(message)
        elif turn["role"] == "tool":
            messages.append({"role": "tool", "tool_call_id": turn["tool_call_id"],
                             "content": turn["text"]})
    out = {"model": model, "messages": messages}
    if norm["max_tokens"] is not None:
        out["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        out["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        out["top_p"] = norm["top_p"]
    if norm["stop"]:
        out["stop"] = norm["stop"]
    if norm["tools"]:
        out["tools"] = [{"type": "function",
                         "function": {"name": tool["name"], "description": tool["description"],
                                      "parameters": tool["parameters"]}}
                        for tool in norm["tools"]]
    choice = norm["tool_choice"]
    if choice is not None:
        if choice["kind"] == "auto":
            out["tool_choice"] = "auto"
        elif choice["kind"] == "any":
            out["tool_choice"] = "required"
        elif choice["kind"] == "none":
            out["tool_choice"] = "none"
        elif choice["kind"] == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if norm["json_mode"]:
        out["response_format"] = {"type": "json_object"}
    if norm["stream"]:
        out["stream"] = True
        # Ask for the usage chunk: without it an OpenAI stream carries no usage.
        out["stream_options"] = {"include_usage": True}
    return out


def to_anthropic(norm, model):
    """The normalized request as an Anthropic Messages body."""
    messages = []
    for turn in norm["turns"]:
        if turn["role"] == "user":
            blocks = []
            for part in turn["parts"]:
                if part["kind"] == "text":
                    blocks.append({"type": "text", "text": part["text"]})
                else:
                    image = part["image"]
                    if "data" in image:
                        blocks.append({"type": "image",
                                       "source": {"type": "base64", "media_type": image["mime"],
                                                  "data": image["data"]}})
                    else:
                        blocks.append({"type": "image",
                                       "source": {"type": "url", "url": image["url"]}})
            if blocks:
                messages.append({"role": "user", "content": blocks})
        elif turn["role"] == "assistant":
            blocks = []
            if turn["text"] != "":
                blocks.append({"type": "text", "text": turn["text"]})
            for call in turn["tool_calls"]:
                try:
                    arguments = json.loads(call["arguments"])
                except Exception:
                    arguments = {}
                blocks.append({"type": "tool_use", "id": call["id"] or "toolu_unknown",
                               "name": call["name"], "input": arguments if is_record(arguments) else {}})
            messages.append({"role": "assistant", "content": blocks})
        elif turn["role"] == "tool":
            messages.append({"role": "user",
                             "content": [{"type": "tool_result", "tool_use_id": turn["tool_call_id"],
                                          "content": turn["text"]}]})
    out = {"model": model, "messages": messages}
    if norm["system"] != "":
        out["system"] = norm["system"]
    if norm["max_tokens"] is not None:
        out["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        out["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        out["top_p"] = norm["top_p"]
    if norm["stop"]:
        out["stop_sequences"] = norm["stop"]
    if norm["tools"]:
        out["tools"] = [{"name": tool["name"], "description": tool["description"],
                         "input_schema": tool["parameters"]} for tool in norm["tools"]]
    choice = norm["tool_choice"]
    if choice is not None:
        if choice["kind"] == "auto":
            out["tool_choice"] = {"type": "auto"}
        elif choice["kind"] == "any":
            out["tool_choice"] = {"type": "any"}
        elif choice["kind"] == "tool":
            out["tool_choice"] = {"type": "tool", "name": choice["name"]}
    if norm["stream"]:
        out["stream"] = True
    return out


def to_gemini(norm, model):
    """The normalized request as a Gemini generateContent body."""
    id_to_name = {}
    for turn in norm["turns"]:
        if turn["role"] == "assistant":
            for call in turn["tool_calls"]:
                if call["id"] != "":
                    id_to_name[call["id"]] = call["name"]
    contents = []
    for turn in norm["turns"]:
        if turn["role"] == "user":
            parts = []
            for part in turn["parts"]:
                if part["kind"] == "text":
                    parts.append({"text": part["text"]})
                else:
                    image = part["image"]
                    if "data" in image:
                        parts.append({"inlineData": {"mimeType": image["mime"], "data": image["data"]}})
                    else:
                        parts.append({"fileData": {"fileUri": image["url"]}})
            if parts:
                contents.append({"role": "user", "parts": parts})
        elif turn["role"] == "assistant":
            parts = []
            if turn["text"] != "":
                parts.append({"text": turn["text"]})
            for call in turn["tool_calls"]:
                try:
                    arguments = json.loads(call["arguments"])
                except Exception:
                    arguments = {}
                parts.append({"functionCall": {"name": call["name"], "args": arguments if is_record(arguments) else {}}})
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif turn["role"] == "tool":
            name = id_to_name.get(turn["tool_call_id"], "")
            contents.append({"role": "function",
                             "parts": [{"functionResponse": {"name": name,
                                                            "response": {"content": turn["text"]}}}]})
    out = {"contents": contents}
    if norm["system"] != "":
        out["systemInstruction"] = {"parts": [{"text": norm["system"]}]}
    if norm["tools"]:
        out["tools"] = [{"functionDeclarations": [
            {"name": tool["name"], "description": tool["description"],
             "parameters": tool["parameters"]} for tool in norm["tools"]]}]
    config = {}
    if norm["max_tokens"] is not None:
        config["maxOutputTokens"] = norm["max_tokens"]
    if norm["stop"]:
        config["stopSequences"] = norm["stop"]
    if norm["json_mode"]:
        config["responseMimeType"] = "application/json"
    if norm["temperature"] is not None:
        config["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        config["topP"] = norm["top_p"]
    if config:
        out["generationConfig"] = config
    return out


# ---------------------------------------------------------------------------
# Provider calls and their answers.
# ---------------------------------------------------------------------------

class ProviderFailure(Exception):
    """A provider refused: its HTTP status and the message it gave."""

    def __init__(self, status, message):
        super().__init__("provider answered HTTP %d: %s" % (status, message))
        self.status = status
        self.message = message


def _error_message(raw):
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return ""
    if is_record(body):
        err = body.get("error")
        if is_record(err) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(body.get("message"), str):
            return body["message"]
    return ""


def call_provider(provider, model, body, stream):
    """One inference call. Returns (status, raw body bytes). Raises on transport."""
    base = join_base(ENV_URL, provider["base"])
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "openai":
        url = base + "/v1/chat/completions"
        if stream:
            headers["Accept"] = "text/event-stream"
    elif provider["wire"] == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        if stream:
            headers["Accept"] = "text/event-stream"
    else:
        action = "streamGenerateContent" if stream else "generateContent"
        url = base + "/v1beta/models/" + urllib.parse.quote(model, safe="") + ":" + action
        if stream:
            url += "?alt=sse"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=PROVIDER_TIMEOUT_S) as resp:
            return (resp.status, resp.read())
    except urllib.error.HTTPError as exc:
        return (exc.code, exc.read())


def _parse_sse_frames(text):
    """SSE text as [(event or None, data string)] frames."""
    frames = []
    for chunk in text.replace("\r\n", "\n").split("\n\n"):
        if chunk.strip() == "":
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


def _openai_text(content):
    if isinstance(content, str):
        return content
    return _text_of(content)


def _gemini_text(parts):
    if not isinstance(parts, list):
        return ""
    texts = []
    for part in parts:
        if is_record(part) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "".join(texts)


def _stop_family(stop):
    """The stop family: 0 stop, 1 length, 2 tool use, 3 refusal; None unknown."""
    families = {"stop": 0, "end_turn": 0, "STOP": 0,
                "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
                "tool_calls": 2, "function_call": 2, "tool_use": 2,
                "content_filter": 3, "refusal": 3, "SAFETY": 3}
    return families.get(stop)


def parse_provider_response(wire, status, raw, stream):
    """A provider's answer as (text, input_tokens, output_tokens, stop).

    Raises ProviderFailure on a refusal, ValueError on a malformed answer.
    """
    if status < 200 or status >= 300:
        raise ProviderFailure(status, _error_message(raw))
    text = raw.decode("utf-8")
    if not stream:
        try:
            body = json.loads(text)
        except Exception:
            raise ValueError("the provider answered a body that is not JSON")
        if not is_record(body):
            raise ValueError("the provider answered a JSON value that is not one object")
        if wire == "openai":
            choices = body.get("choices") if isinstance(body.get("choices"), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else {}
            message = choice.get("message") if is_record(choice.get("message")) else {}
            usage = body.get("usage") if is_record(body.get("usage")) else {}
            return (_openai_text(message.get("content")),
                    num_or_none(usage.get("prompt_tokens")) or 0,
                    num_or_none(usage.get("completion_tokens")) or 0,
                    str_of(choice.get("finish_reason")))
        if wire == "anthropic":
            usage = body.get("usage") if is_record(body.get("usage")) else {}
            return (_text_of(body.get("content")),
                    num_or_none(usage.get("input_tokens")) or 0,
                    num_or_none(usage.get("output_tokens")) or 0,
                    str_of(body.get("stop_reason")))
        candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
        candidate = candidates[0] if candidates and is_record(candidates[0]) else {}
        content = candidate.get("content") if is_record(candidate.get("content")) else {}
        usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
        return (_gemini_text(content.get("parts")),
                num_or_none(usage.get("promptTokenCount")) or 0,
                num_or_none(usage.get("candidatesTokenCount")) or 0,
                str_of(candidate.get("finishReason")))
    # Streaming: read the whole stream, then reassemble it.
    frames = _parse_sse_frames(text)
    if wire == "openai":
        answer = ""
        stop = ""
        usage_in = 0
        usage_out = 0
        for _event, data in frames:
            if data.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get("usage")):
                usage_in = num_or_none(chunk["usage"].get("prompt_tokens")) or usage_in
                usage_out = num_or_none(chunk["usage"].get("completion_tokens")) or usage_out
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            if not choices or not is_record(choices[0]):
                continue
            choice = choices[0]
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            if isinstance(delta.get("content"), str):
                answer += delta["content"]
            if isinstance(choice.get("finish_reason"), str):
                stop = choice["finish_reason"]
        return (answer, usage_in, usage_out, stop)
    if wire == "anthropic":
        answer = ""
        stop = ""
        usage_in = 0
        usage_out = 0
        for event, data in frames:
            try:
                payload = json.loads(data)
            except Exception:
                continue
            if not is_record(payload):
                continue
            if event == "message_start" and is_record(payload.get("message")):
                usage = payload["message"].get("usage")
                if is_record(usage):
                    usage_in = num_or_none(usage.get("input_tokens")) or usage_in
            elif event == "content_block_delta" and is_record(payload.get("delta")):
                if payload["delta"].get("type") == "text_delta" and isinstance(payload["delta"].get("text"), str):
                    answer += payload["delta"]["text"]
            elif event == "message_delta":
                if is_record(payload.get("delta")) and isinstance(payload["delta"].get("stop_reason"), str):
                    stop = payload["delta"]["stop_reason"]
                if is_record(payload.get("usage")):
                    usage_out = num_or_none(payload["usage"].get("output_tokens")) or usage_out
        return (answer, usage_in, usage_out, stop)
    answer = ""
    stop = ""
    usage_in = 0
    usage_out = 0
    for _event, data in frames:
        try:
            payload = json.loads(data)
        except Exception:
            continue
        if not is_record(payload):
            continue
        candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
        if candidates and is_record(candidates[0]):
            candidate = candidates[0]
            content = candidate.get("content") if is_record(candidate.get("content")) else {}
            answer += _gemini_text(content.get("parts"))
            if isinstance(candidate.get("finishReason"), str):
                stop = candidate["finishReason"]
        usage = payload.get("usageMetadata")
        if is_record(usage):
            usage_in = num_or_none(usage.get("promptTokenCount")) or usage_in
            usage_out = num_or_none(usage.get("candidatesTokenCount")) or usage_out
    return (answer, usage_in, usage_out, stop)


# ---------------------------------------------------------------------------
# Answers in the wire the client used.
# ---------------------------------------------------------------------------

def _split_text(text):
    """13-character pieces, as the providers split their streams; [""] when empty."""
    if text == "":
        return [""]
    return [text[i:i + 13] for i in range(0, len(text), 13)]


def _openai_stop(stop):
    family = _stop_family(stop)
    return {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}.get(family, "stop")


def _anthropic_stop(stop):
    family = _stop_family(stop)
    return {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}.get(family, "end_turn")


def openai_client_json(text, usage_in, usage_out, stop, model):
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_OPENAI_CREATED,
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": _openai_stop(stop),
                     "logprobs": None}],
        "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                  "total_tokens": usage_in + usage_out},
    }


def _sse_data(payload):
    return "data: %s\n\n" % json.dumps(payload)


def openai_client_sse(text, usage_in, usage_out, stop, model):
    def chunk(delta, finish):
        return {"id": "chatcmpl-sim", "object": "chat.completion.chunk",
                "created": FIXED_OPENAI_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    frames = [_sse_data(chunk({"role": "assistant", "content": ""}, None))]
    for piece in _split_text(text):
        frames.append(_sse_data(chunk({"content": piece}, None)))
    frames.append(_sse_data(chunk({}, _openai_stop(stop))))
    frames.append(_sse_data({"id": "chatcmpl-sim", "object": "chat.completion.chunk",
                             "created": FIXED_OPENAI_CREATED, "model": model, "choices": [],
                             "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                                       "total_tokens": usage_in + usage_out}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_client_json(text, usage_in, usage_out, stop, model):
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": _anthropic_stop(stop),
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
    }


def anthropic_client_sse(text, usage_in, usage_out, stop, model):
    def event(name, payload):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(payload))
    frames = [event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_sim", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage_in, "output_tokens": 0}}})]
    frames.append(event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in _split_text(text):
        frames.append(event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _anthropic_stop(stop), "stop_sequence": None},
        "usage": {"output_tokens": usage_out}}))
    frames.append(event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_error(message, error_type="invalid_request_error", code=None):
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


# ---------------------------------------------------------------------------
# Serving one client request.
# ---------------------------------------------------------------------------

class ServerState:
    """The brief, the probes, the catalog, the router and the arrival count."""

    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.probes = probes
        self.catalog = catalog
        self.router = RouterState()
        self.manifest_index = 0
        self.lock = threading.Lock()

    def requirements_for(self, body, client_api):
        """The manifest row by arrival order, else the body's own requirements."""
        manifest = []
        if self.brief["workload"] is not None:
            manifest = self.brief["workload"]["manifest"]
        with self.lock:
            index = self.manifest_index
            self.manifest_index += 1
        if index < len(manifest):
            return manifest[index]["requires"]
        # Fallback: read the requirements off the body itself.
        if client_api == "openai":
            norm = normalize_openai(body)
        else:
            norm = normalize_anthropic(body)
        requires = {}
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        vision = False
        for raw in messages:
            if not is_record(raw) or not isinstance(raw.get("content"), list):
                continue
            for block in raw["content"]:
                if is_record(block) and block.get("type") in ("image", "image_url"):
                    vision = True
        if norm["tools"]:
            requires["tools"] = True
        if vision:
            requires["vision"] = True
        if norm["json_mode"]:
            requires["json"] = True
        return requires


def _upstream_body(client_api, norm, body, provider, model):
    """The request in the provider's wire.

    Same-wire requests pass the client's body through with only the model
    replaced (zero translation risk); cross-wire requests are rebuilt from
    the normalized form. An OpenAI stream always asks for the usage chunk.
    """
    wire = provider["wire"]
    if client_api == "openai" and wire == "openai":
        out = dict(body)
        out["model"] = model
        if body.get("stream") is True:
            out["stream_options"] = {"include_usage": True}
        return out
    if client_api == "anthropic" and wire == "anthropic":
        out = dict(body)
        out["model"] = model
        return out
    if wire == "openai":
        return to_openai(norm, model)
    if wire == "anthropic":
        return to_anthropic(norm, model)
    return to_gemini(norm, model)


def serve_request(state, client_api, body):
    """Answer one client request. Returns (status, content_type, body_bytes)."""
    if not is_record(body):
        if client_api == "anthropic":
            return (400, "application/json", json.dumps(anthropic_error("invalid_request_error", "the request body is not one JSON object")).encode("utf-8"))
        return (400, "application/json", json.dumps(openai_error("the request body is not one JSON object")).encode("utf-8"))
    norm = normalize_openai(body) if client_api == "openai" else normalize_anthropic(body)
    requires = state.requirements_for(body, client_api)
    stream = norm["stream"]
    candidates = candidates_for(state.probes, state.brief["ratings"], requires,
                                norm["json_mode"], state.router)
    if not candidates:
        message = "no free model on the directory meets this request"
        if client_api == "anthropic":
            return (503, "application/json", json.dumps(anthropic_error("overloaded_error", message)).encode("utf-8"))
        return (503, "application/json", json.dumps(openai_error(message, "no_free_model")).encode("utf-8"))
    last_error = None
    for candidate in candidates[:MAX_ATTEMPTS_PER_REQUEST]:
        provider = candidate["provider"]
        model = candidate["model"]["id"]
        upstream = _upstream_body(client_api, norm, body, provider, model)
        state.router.record_call(provider["id"])
        try:
            status, raw = call_provider(provider, model, upstream, stream)
            text, usage_in, usage_out, stop = parse_provider_response(provider["wire"], status, raw, stream)
        except ProviderFailure as exc:
            state.router.record_failure(provider["id"], exc.message)
            last_error = "the provider answered HTTP %d" % exc.status
            log("%s/%s refused (%s); failing over" % (provider["id"], model, exc.message or exc.status))
            continue
        except ValueError as exc:
            state.router.record_failure(provider["id"], "")
            last_error = str(exc)
            log("%s/%s answered malformed (%s); failing over" % (provider["id"], model, exc))
            continue
        except Exception as exc:
            state.router.record_failure(provider["id"], "")
            last_error = str(exc)
            log("%s/%s unreachable (%s); failing over" % (provider["id"], model, exc))
            continue
        state.router.record_success(provider["id"])
        if client_api == "openai":
            if stream:
                payload = openai_client_sse(text, usage_in, usage_out, stop, model)
                return (200, "text/event-stream; charset=utf-8", payload.encode("utf-8"))
            payload = openai_client_json(text, usage_in, usage_out, stop, model)
            return (200, "application/json", json.dumps(payload).encode("utf-8"))
        if stream:
            payload = anthropic_client_sse(text, usage_in, usage_out, stop, model)
            return (200, "text/event-stream; charset=utf-8", payload.encode("utf-8"))
        payload = anthropic_client_json(text, usage_in, usage_out, stop, model)
        return (200, "application/json", json.dumps(payload).encode("utf-8"))
    message = "every free model refused this request" + (": %s" % last_error if last_error else "")
    if client_api == "anthropic":
        return (503, "application/json", json.dumps(anthropic_error("overloaded_error", message)).encode("utf-8"))
    return (503, "application/json", json.dumps(openai_error(message, "no_free_model")).encode("utf-8"))


# ---------------------------------------------------------------------------
# The HTTP server, the register and the work poll.
# ---------------------------------------------------------------------------

def _models_response(state, req):
    """GET /v1/models, in the shape of the auth header it got."""
    models = served_models(state.probes, state.brief["ratings"])
    if req.headers.get("x-api-key") is not None:
        body = {
            "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                      "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
            "has_more": False,
        }
    else:
        body = {
            "object": "list",
            "data": [{"id": m["id"], "object": "model", "created": FIXED_OPENAI_CREATED,
                      "owned_by": m["provider"]} for m in models],
        }
    return (200, "application/json", json.dumps(body).encode("utf-8"))


class AggregatorHandler(BaseHTTPRequestHandler):
    """The entry's server. `state` is set by start() before serving."""

    state = None
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def _send(self, status, content_type, payload):
        # Always send Content-Length: the client reads SSE until the body
        # ends, and without a length it would wait on the kept-alive
        # connection forever (the whole SSE payload is buffered first).
        headers = {"Content-Type": content_type,
                   "Content-Length": str(len(payload))}
        if content_type != "application/json":
            headers["Cache-Control"] = "no-cache"
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _send_error(self, status, client_api, message):
        if client_api == "anthropic":
            payload = json.dumps(anthropic_error("invalid_request_error", message)).encode("utf-8")
        else:
            payload = json.dumps(openai_error(message)).encode("utf-8")
        self._send(status, "application/json", payload)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/models":
            status, content_type, payload = _models_response(self.state, self)
            self._send(status, content_type, payload)
            return
        self._send_error(404, "openai", "no route GET %s" % path)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/chat/completions":
            body = self._read_json()
            if body is None:
                self._send_error(400, "openai", "the request body is not one JSON object")
                return
            try:
                status, content_type, payload = serve_request(self.state, "openai", body)
            except Exception as exc:
                log("chat completions failed: %s" % exc)
                self._send_error(500, "openai", "the server failed to answer")
                return
            self._send(status, content_type, payload)
            return
        if path == "/v1/messages":
            body = self._read_json()
            if body is None:
                self._send_error(400, "anthropic", "the request body is not one JSON object")
                return
            try:
                status, content_type, payload = serve_request(self.state, "anthropic", body)
            except Exception as exc:
                log("messages failed: %s" % exc)
                self._send_error(500, "anthropic", "the server failed to answer")
                return
            self._send(status, content_type, payload)
            return
        self._send_error(404, "openai", "no route POST %s" % path)


def register(port, catalog):
    """POST /register with the port and the one catalog; retry while the env wakes."""
    body = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}).encode("utf-8")
    last_error = None
    for _attempt in range(10):
        try:
            req = urllib.request.Request(
                ENV_URL + "/register", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:
                reply = json.loads(resp.read().decode("utf-8"))
            if resp.status == 200 and is_record(reply):
                log("registered: the workload holds %s requests" % reply.get("requests"))
                return
            last_error = "HTTP %d" % resp.status
        except urllib.error.HTTPError as exc:
            last_error = "HTTP %d" % exc.code
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError("POST %s/register failed: %s" % (ENV_URL, last_error))


def poll_work():
    """Poll GET /work; exit 0 the moment the env says the workload is done."""
    while True:
        try:
            req = urllib.request.Request(ENV_URL + "/work", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    work = json.loads(resp.read().decode("utf-8"))
                    if is_record(work) and work.get("done") is True:
                        log("the workload is done; exiting 0")
                        os._exit(0)
        except SystemExit:
            raise
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)


def start():
    brief = read_brief(PROMPT_FILE)
    if brief["error"] is not None:
        log(brief["error"])
    probes = []
    for provider in brief["providers"]:
        ok, status, models, error = probe_provider(ENV_URL, provider)
        if not ok:
            # One retry: the env may still be waking.
            time.sleep(0.2)
            ok, status, models, error = probe_provider(ENV_URL, provider)
        if not ok and error is not None:
            log(error)
        probes.append({"provider": provider, "ok": ok, "status": status, "models": models})
    catalog = build_catalog(probes, brief["ratings"])
    if "--list" in sys.argv:
        sys.stdout.write(json.dumps(catalog) + "\n")
        sys.stdout.flush()
        sys.exit(0)
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    state = ServerState(brief, probes, catalog)
    AggregatorHandler.state = state
    server = ThreadingHTTPServer(("0.0.0.0", 0), AggregatorHandler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len(catalog["providers"]),
           len(served_models(probes, brief["ratings"]))))
    register(port, catalog)
    thread = threading.Thread(target=poll_work, daemon=True)
    thread.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    try:
        start()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        log(str(exc))
        sys.exit(1)
