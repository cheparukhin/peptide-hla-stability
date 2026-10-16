#!/usr/bin/env python3
"""Generic LLM subagent fan-out harness (Anthropic Messages API).

Runs N independent prompts concurrently as "subagents", each with the same
system prompt and its own task prompt, and collects the results to JSONL.

This is a *reasoning* fan-out tool: use it for work that needs a language model
(literature sweeps, report audits, evidence gathering). It is NOT an experiment
runner -- numerical sweeps over the Phase A/C/D fits need no LLM and belong in
multiprocessing or Modal (see HACKATHON_PLAN.md).

Cost discipline (per the project's standing rule: price every paid run first):

  * `--dry-run` is the default. It prices the run with `messages.count_tokens`,
    which is a free API call, and exits without spending anything.
  * Spending requires `--go`. There is no way to bill the account by accident.
  * Every result row carries its own token usage and dollar cost, and the run
    prints a total.

Credentials are resolved by the SDK itself, in this order: ANTHROPIC_API_KEY,
then ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile under
~/.config/anthropic/. This script additionally loads ~/.env and ./.env first so
a key kept there is picked up. It never prints or logs a credential.

Usage
-----
    # 1. Price it (free, spends nothing)
    .venv/bin/python scripts/fanout.py --tasks tasks.jsonl --system system.md

    # 2. Run it
    .venv/bin/python scripts/fanout.py --tasks tasks.jsonl --system system.md \
        --out reports/further_research/sweep.jsonl --go

    # Research subagents with web search, structured results
    .venv/bin/python scripts/fanout.py --tasks questions.jsonl \
        --system system.md --web-search --schema finding.schema.json \
        --out out.jsonl --go

Task file format -- JSONL (one object per line) or a JSON array:

    {"id": "phase_c_leakage", "prompt": "Assess whether ..."}
    {"id": "assay_floor",     "prompt": "Summarise the evidence that ..."}

An optional per-task "system" key overrides the shared system prompt, and an
optional "max_tokens" overrides the default for that task alone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("anthropic SDK missing. Install it with:\n"
             "  uv pip install --python .venv/bin/python anthropic python-dotenv")

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # optional
    load_dotenv = None


# ---------------------------------------------------------------------------
# Pricing, USD per million tokens. Cached from the Anthropic pricing page as of
# 2026-09-25 -- re-check https://platform.claude.com/ before quoting these in a
# report, and treat any estimate below as an estimate.
# ---------------------------------------------------------------------------
PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5":   {"in": 4.00, "out": 20.00, "cache_read": 0.20},
    "claude-opus-5":     {"in": 5.00, "out": 25.00, "cache_read": 0.50},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.20},
    "claude-haiku-4-5":  {"in": 1.00, "out":  5.00, "cache_read": 0.10},
    "claude-fable-5-1":  {"in": 10.00, "out": 50.00, "cache_read": 0.25},
}
CACHE_WRITE_MULTIPLIER = 1.25  # writing to cache costs ~1.25x the input rate

# Models on which the server-side refusal fallback is available (Claude API).
FALLBACK_MODELS = {
    "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1",
}


def price_usd(model: str, usage: Any) -> float:
    """Dollar cost of one response from its reported usage."""
    p = PRICES.get(model)
    if p is None:
        return float("nan")
    uncached = getattr(usage, "input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    return (
        uncached * p["in"]
        + cache_write * p["in"] * CACHE_WRITE_MULTIPLIER
        + cache_read * p["cache_read"]
        + out * p["out"]
    ) / 1_000_000


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
CREDENTIAL_HELP = """\
No usable Anthropic credential was found.

The SDK looks for, in order: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, then an
OAuth profile under ~/.config/anthropic/. A profile *config* can exist with no
credentials behind it, which is its own error, not a missing profile.

To store an API key without it passing through a shell history or a transcript
(your typing stays hidden). ~/.env sits outside this repository, so git never
sees it; a repo-local .env works too and is already in .gitignore:

  printf "Enter ANTHROPIC_API_KEY (typing hidden): " && read -rs val \\
    && echo && printf 'ANTHROPIC_API_KEY=%s\\n' "$val" >> ~/.env \\
    && unset val && echo "Saved to ~/.env"

Create a key at https://platform.claude.com/ -> API keys. Alternatively, run
`ant auth login` to populate the OAuth profile the SDK already expects."""


def preflight_auth(client: anthropic.Anthropic) -> None:
    """Fail fast and once on a credential problem.

    `models.list()` is a free call -- it bills no tokens -- so this costs
    nothing and avoids printing the same auth failure N times, once per task.
    """
    try:
        client.models.list()
    except (anthropic.CredentialsError, anthropic.AuthenticationError,
            anthropic.PermissionDeniedError) as exc:
        sys.exit(f"{type(exc).__name__}: {exc}\n\n{CREDENTIAL_HELP}")
    except anthropic.APIConnectionError as exc:
        sys.exit(f"Cannot reach the Anthropic API: {exc}")


def load_tasks(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        sys.exit(f"{path}: empty task file")

    if raw.lstrip().startswith("["):
        tasks = json.loads(raw)
    else:
        tasks = []
        for lineno, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{lineno}: not valid JSON -- {exc}")

    seen: set[str] = set()
    for i, t in enumerate(tasks):
        if not isinstance(t, dict) or "prompt" not in t:
            sys.exit(f"{path}: task {i} needs at least a 'prompt' key")
        t.setdefault("id", f"task_{i:04d}")
        if t["id"] in seen:
            sys.exit(f"{path}: duplicate task id {t['id']!r} -- ids must be unique "
                     "so --resume can tell rows apart")
        seen.add(t["id"])
    return tasks


def already_done(out_path: Path) -> set[str]:
    """Task ids with a successful row already in the output file."""
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ok"):
            done.add(row.get("id"))
    return done


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------
def build_request(task: dict[str, Any], cfg: argparse.Namespace,
                  schema: dict | None) -> dict[str, Any]:
    system_text = task.get("system") or cfg.system_text

    # Stable prefix (system) before volatile content (the task prompt) so the
    # cached prefix is shared across every subagent in the fan-out. Caching only
    # pays off once the system prompt clears the model's minimum cacheable
    # prefix (512-4096 tokens depending on model); below that it is a no-op.
    system: Any = None
    if system_text:
        block: dict[str, Any] = {"type": "text", "text": system_text}
        if cfg.cache_system:
            block["cache_control"] = {"type": "ephemeral"}
        system = [block]

    output_config: dict[str, Any] = {"effort": cfg.effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}

    req: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": int(task.get("max_tokens") or cfg.max_tokens),
        "messages": [{"role": "user", "content": task["prompt"]}],
        "output_config": output_config,
        # Thinking is always on for the Opus 5.x / Sonnet 5.5 family; `effort`
        # above is the depth control. `summarized` costs nothing extra -- the
        # reasoning is billed identically whether or not it is returned -- and
        # it is worth keeping for provenance.
        "thinking": {"type": "adaptive", "display": cfg.thinking_display},
    }
    if system is not None:
        req["system"] = system
    if cfg.web_search:
        req["tools"] = [{"type": "web_search_20260209", "name": "web_search",
                         "max_uses": cfg.max_web_uses}]
    return req


# ---------------------------------------------------------------------------
# One subagent
# ---------------------------------------------------------------------------
async def run_task(client: AsyncAnthropic, sem: asyncio.Semaphore,
                   task: dict[str, Any], cfg: argparse.Namespace,
                   schema: dict | None) -> dict[str, Any]:
    req = build_request(task, cfg, schema)
    started = time.monotonic()

    async with sem:
        try:
            response, continuations = await _call_with_pause_resume(client, req, cfg)
        except (anthropic.BadRequestError, anthropic.NotFoundError,
                anthropic.PermissionDeniedError, anthropic.AuthenticationError) as exc:
            # Not retryable: the request or the credential is wrong, and
            # re-running it unchanged will fail the same way.
            return _error_row(task, cfg, type(exc).__name__,
                              getattr(exc, "message", str(exc)), started,
                              retryable=False)
        except (anthropic.RateLimitError, anthropic.APITimeoutError) as exc:
            # The SDK already retried these per --max-retries and still failed;
            # worth another pass later via --resume.
            return _error_row(task, cfg, type(exc).__name__,
                              getattr(exc, "message", str(exc)), started,
                              retryable=True)
        except anthropic.APIStatusError as exc:
            status = getattr(exc, "status_code", 0) or 0
            return _error_row(task, cfg, type(exc).__name__,
                              getattr(exc, "message", str(exc)), started,
                              retryable=status >= 500)
        except anthropic.APIConnectionError as exc:
            return _error_row(task, cfg, "APIConnectionError", str(exc), started,
                              retryable=True)
        except Exception as exc:  # pragma: no cover - last resort
            return _error_row(task, cfg, type(exc).__name__, str(exc)[:500],
                              started, retryable=False)

    elapsed = time.monotonic() - started

    # Guard stop_reason before reading content. A refusal is HTTP 200 with no
    # usable answer, so it must not be recorded as a success.
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        row = _error_row(task, cfg, "refusal",
                         getattr(details, "explanation", "") or "model declined",
                         started)
        row["refusal_category"] = getattr(details, "category", None)
        row["cost_usd"] = round(price_usd(response.model, response.usage), 6)
        return row

    text = "\n".join(b.text for b in response.content
                     if b.type == "text" and getattr(b, "text", ""))
    thinking = "\n".join(getattr(b, "thinking", "") for b in response.content
                         if b.type == "thinking" and getattr(b, "thinking", ""))

    parsed: Any = None
    parse_error: str | None = None
    if schema is not None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            parse_error = f"schema was requested but output is not JSON: {exc}"

    u = response.usage
    return {
        "id": task["id"],
        "ok": parse_error is None,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "truncated": response.stop_reason == "max_tokens",
        "continuations": continuations,
        "text": text,
        "parsed": parsed,
        "parse_error": parse_error,
        "thinking": thinking or None,
        "usage": {
            "input_tokens": getattr(u, "input_tokens", 0),
            "output_tokens": getattr(u, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0),
        },
        "cost_usd": round(price_usd(response.model, u), 6),
        "elapsed_s": round(elapsed, 2),
        "request_id": getattr(response, "_request_id", None),
    }


async def _call_with_pause_resume(client: AsyncAnthropic, req: dict[str, Any],
                                  cfg: argparse.Namespace):
    """Issue the request, resuming across `pause_turn` for server tools.

    Web search can stop a turn with stop_reason "pause_turn" rather than
    finishing. That is not an error -- the turn is resumed by re-sending it with
    the partial assistant content appended.
    """
    messages = list(req["messages"])
    use_fallbacks = cfg.fallbacks and req["model"] in FALLBACK_MODELS

    for continuations in range(cfg.max_continuations + 1):
        call = dict(req, messages=messages)
        if use_fallbacks:
            try:
                response = await client.beta.messages.create(
                    **call,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except anthropic.BadRequestError:
                # Account or model does not accept the fallback beta; drop it
                # for the rest of the run rather than failing every task.
                use_fallbacks = False
                cfg.fallbacks = False
                response = await client.messages.create(**call)
        else:
            response = await client.messages.create(**call)

        if response.stop_reason != "pause_turn":
            return response, continuations
        messages = [*messages, {"role": "assistant", "content": response.content}]

    return response, cfg.max_continuations


def _error_row(task: dict[str, Any], cfg: argparse.Namespace, kind: str,
               message: str, started: float,
               retryable: bool = False) -> dict[str, Any]:
    return {
        "id": task["id"],
        "ok": False,
        "model": cfg.model,
        "error": kind,
        "error_message": message[:1000],
        "retryable": retryable,
        "text": None,
        "cost_usd": 0.0,
        "elapsed_s": round(time.monotonic() - started, 2),
    }


# ---------------------------------------------------------------------------
# Dry run: price the fan-out without spending anything
# ---------------------------------------------------------------------------
def dry_run(client: anthropic.Anthropic, tasks: list[dict[str, Any]],
            cfg: argparse.Namespace, schema: dict | None) -> None:
    print(f"Pricing {len(tasks)} task(s) on {cfg.model} "
          f"(effort={cfg.effort}) with messages.count_tokens -- this is free.\n")

    total_in = 0
    failures = 0
    per_task: list[tuple[str, int]] = []
    for task in tasks:
        req = build_request(task, cfg, schema)
        try:
            counted = client.messages.count_tokens(
                model=req["model"],
                messages=req["messages"],
                **({"system": req["system"]} if "system" in req else {}),
                **({"tools": req["tools"]} if "tools" in req else {}),
            )
        except Exception as exc:
            failures += 1
            print(f"  ! {task['id']}: could not count tokens ({type(exc).__name__})")
            continue
        total_in += counted.input_tokens
        per_task.append((task["id"], counted.input_tokens))

    if failures and not per_task:
        sys.exit("\nCould not price any task -- check credentials (see --help).")

    p = PRICES.get(cfg.model)
    assumed_out = cfg.assume_output_tokens * len(per_task)

    for tid, n in per_task[:10]:
        print(f"  {tid:<32} {n:>8,} input tokens")
    if len(per_task) > 10:
        print(f"  ... and {len(per_task) - 10} more")

    print(f"\n  input tokens (measured)        {total_in:>12,}")
    print(f"  output tokens (ASSUMED {cfg.assume_output_tokens:,}/task) {assumed_out:>12,}")

    if p is None:
        print(f"\n  No cached price for {cfg.model}; cannot estimate cost.")
    else:
        cost_in = total_in * p["in"] / 1e6
        cost_out = assumed_out * p["out"] / 1e6
        print(f"\n  estimated input cost           ${cost_in:>11.4f}")
        print(f"  estimated output cost          ${cost_out:>11.4f}")
        print(f"  ESTIMATED TOTAL                ${cost_in + cost_out:>11.4f}")
        if cfg.cache_system and cfg.system_text:
            print("\n  Note: the shared system prompt is cached, so repeat runs and the")
            print("  tasks after the first pay ~0.1x on that prefix. The figure above")
            print("  does not assume any cache hit, so it is an upper bound on input.")
        if cfg.web_search:
            print("\n  Note: --web-search adds per-search charges and extra input tokens")
            print("  for fetched content that count_tokens cannot see. Treat the figure")
            print("  above as a floor, not a ceiling.")

    print("\nOutput tokens are an assumption, not a measurement -- tune with")
    print("--assume-output-tokens. Nothing has been spent. Re-run with --go to execute.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def fan_out(tasks: list[dict[str, Any]], cfg: argparse.Namespace,
                  schema: dict | None, out_path: Path) -> list[dict[str, Any]]:
    client = AsyncAnthropic(max_retries=cfg.max_retries)
    sem = asyncio.Semaphore(cfg.concurrency)
    rows: list[dict[str, Any]] = []
    write_lock = asyncio.Lock()

    out_fh = out_path.open("a", encoding="utf-8")

    async def one(task: dict[str, Any]) -> None:
        row = await run_task(client, sem, task, cfg, schema)
        async with write_lock:
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fh.flush()  # partial progress survives a crash or Ctrl-C
            rows.append(row)
            status = "ok " if row.get("ok") else "FAIL"
            note = "" if row.get("ok") else f"  <- {row.get('error', '')}"
            print(f"  [{len(rows):>3}/{len(tasks)}] {status} {row['id']:<32} "
                  f"{row['elapsed_s']:>6.1f}s  ${row.get('cost_usd', 0):.4f}{note}",
                  flush=True)

    try:
        pending = list(tasks)
        # Warm the shared cache with one task before fanning out: concurrent
        # first requests would all miss the cache and each pay to write it.
        if cfg.cache_system and cfg.system_text and len(pending) > 1:
            print("  warming the shared system-prompt cache with one task first...")
            await one(pending.pop(0))
        await asyncio.gather(*(one(t) for t in pending))
    finally:
        out_fh.close()
        await client.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fan out N LLM subagents over a task file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----")[-1],
    )
    ap.add_argument("--tasks", required=True, type=Path,
                    help="JSONL or JSON-array file of {id, prompt} objects")
    ap.add_argument("--system", type=Path,
                    help="file holding the shared system prompt")
    ap.add_argument("--out", type=Path,
                    help="JSONL output path (default: alongside --tasks)")
    ap.add_argument("--schema", type=Path,
                    help="JSON Schema file; makes each subagent return validated JSON")

    ap.add_argument("--model", default="claude-opus-5-5",
                    help="default: claude-opus-5-5. claude-sonnet-5-5 is the "
                         "usual cost lever for wide fan-outs (~half the price)")
    ap.add_argument("--effort", default="medium",
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="thinking depth / token spend (default: medium)")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--thinking-display", default="summarized",
                    choices=["summarized", "omitted"],
                    help="summarized costs no extra and preserves provenance")

    ap.add_argument("--concurrency", type=int, default=5,
                    help="subagents in flight at once (default: 5)")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="SDK-level retries for 429/5xx (default: 2)")
    ap.add_argument("--web-search", action="store_true",
                    help="give each subagent the server-side web_search tool")
    ap.add_argument("--max-web-uses", type=int, default=5)
    ap.add_argument("--max-continuations", type=int, default=5,
                    help="how many times to resume a paused server-tool turn")
    ap.add_argument("--no-cache-system", dest="cache_system", action="store_false",
                    help="disable prompt caching of the shared system prompt")
    ap.add_argument("--no-fallbacks", dest="fallbacks", action="store_false",
                    help="disable the server-side refusal fallback")

    ap.add_argument("--assume-output-tokens", type=int, default=2000,
                    help="output tokens per task assumed when pricing (default: 2000)")
    ap.add_argument("--resume", action="store_true",
                    help="skip task ids already recorded ok in --out")
    ap.add_argument("--go", action="store_true",
                    help="actually spend money. Without this the run is priced only.")
    cfg = ap.parse_args()

    # Load .env files so a key kept there is picked up. Never printed.
    if load_dotenv is not None:
        load_dotenv(Path.home() / ".env", override=False)
        load_dotenv(Path(".env"), override=False)

    tasks = load_tasks(cfg.tasks)
    cfg.system_text = cfg.system.read_text(encoding="utf-8") if cfg.system else ""
    schema = json.loads(cfg.schema.read_text(encoding="utf-8")) if cfg.schema else None

    out_path = cfg.out or cfg.tasks.with_suffix(".results.jsonl")

    if cfg.resume:
        done = already_done(out_path)
        if done:
            before = len(tasks)
            tasks = [t for t in tasks if t["id"] not in done]
            print(f"--resume: skipping {before - len(tasks)} task(s) already done\n")
    if not tasks:
        print("Nothing to do.")
        return

    probe = anthropic.Anthropic()
    preflight_auth(probe)

    if not cfg.go:
        dry_run(probe, tasks, cfg, schema)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(tasks)} subagent(s) on {cfg.model} "
          f"(effort={cfg.effort}, concurrency={cfg.concurrency})")
    print(f"Appending results to {out_path}\n")

    started = time.monotonic()
    rows = asyncio.run(fan_out(tasks, cfg, schema, out_path))

    ok = [r for r in rows if r.get("ok")]
    total = sum(r.get("cost_usd") or 0.0 for r in rows)
    truncated = [r["id"] for r in rows if r.get("truncated")]

    print(f"\n{len(ok)}/{len(rows)} succeeded in {time.monotonic() - started:.1f}s")
    print(f"ACTUAL COST  ${total:.4f}")
    if truncated:
        print(f"\nTruncated at max_tokens (answers are cut off): {', '.join(truncated)}")
        print("Re-run those with a larger --max-tokens.")
    failures = [r for r in rows if not r.get("ok")]
    for r in failures:
        mark = "retryable" if r.get("retryable") else "permanent"
        print(f"  FAILED [{mark}] {r['id']}: {r.get('error')} "
              f"{(r.get('error_message') or '')[:160]}")
    if failures:
        if any(r.get("retryable") for r in failures):
            print("\nRe-run with --resume to retry only the failures.")
        else:
            print("\nEvery failure is permanent -- --resume will not help. Fix the "
                  "request or the credential first.")


if __name__ == "__main__":
    main()
