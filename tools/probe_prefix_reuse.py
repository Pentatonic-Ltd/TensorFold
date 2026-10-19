"""Read-only probes of TensorFold's prompt-prefix reuse: does a partial match reuse anything, and what survives churn?

Run against a serving endpoint (never starts, stops or loads one):

  python3 tools/probe_prefix_reuse.py http://10.99.0.1:8890/v1 qwen3.6

Every request is greedy with ``max_tokens`` 1, so a phase costs one prompt prefill. The counts printed are the
server's own usage (``prompt_tokens``, ``prompt_tokens_details.cached_tokens``): nothing is inferred from text.

Phases:
  reuse  two requests with the same system message and different user text (whole-block reuse), then one whose
         system message shares the leading block and differs after it (partial match).
  churn  a twice-used block, then distinct single-use prompts, then the block again (retention under turnover).

Report of what these measured on the production build: docs/prefix-cache-retention.md.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

LINE = "The archivist records each page in the ledger, in order, and notes the hand that wrote it. "


def ask(base: str, model: str, system: str, user: str, label: str) -> int:
    """One chat completion; print its server-reported prompt and cached token counts; return the cached count."""

    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 1, "temperature": 0}
    request = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                    headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.load(response)
    wall = (time.perf_counter() - started) * 1000
    usage = reply.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    print(f"{label:32s} prompt_tokens={usage.get('prompt_tokens', 0):7d} "
          f"cached_tokens={cached:7d} wall={wall:7.0f} ms", flush=True)
    return cached


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", nargs="?", default="http://127.0.0.1:8080/v1", help="the endpoint's /v1 URL")
    parser.add_argument("model", nargs="?", default="local-model", help="the model id the server answers to")
    parser.add_argument("--phase", choices=("reuse", "churn", "both"), default="both")
    args = parser.parse_args()

    stable = LINE * 60
    system_a = stable + "RULESET A: the shared chat prefix ends here.\n"
    system_b = stable + "RULESET B: a different deployment writes other words in this same place.\n"

    if args.phase in ("reuse", "both"):
        print("-- reuse: whole message vs shared leading block")
        ask(args.base, args.model, system_a, "probe one", "A, user one (cold)")
        ask(args.base, args.model, system_a, "probe two", "A, user two (same system)")
        ask(args.base, args.model, system_b, "probe three", "B, user three (B shares its head)")

    if args.phase in ("churn", "both"):
        print("-- churn: does a twice-used block survive distinct prompts?")
        ask(args.base, args.model, system_a, "hold", "A before (touch the block)")
        for n in range(8):
            ask(args.base, args.model, LINE * 14 + f"unrelated server {n} writes its own rules here.\n",
                f"other {n}", f"cold {n}")
        hit = ask(args.base, args.model, system_a, "after", "A after 8 cold prompts")
        print("block still resident" if hit else "block was evicted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
