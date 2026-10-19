# Keeping the shared chat prefix hot: the prefix cache on the production Qwen3.6 build

Investigation for SEE-5083 (epic SEE-5107). Read-only: no box, container, service or deploy was changed; the
only writes were the probes in §3 against `:8890`, and every count below is the server's own usage report.
Measured 19 Oct 2026, 09:12–09:16 UTC.

| Claim | Where it comes from |
| --- | --- |
| policy, points, pool, knobs | the deployed source in the container, read-only (§1–§2) |
| reuse vs partial match, retention under churn | 17 requests against production (§3) |
| per-slot costs and headroom | computed from `cuda/geometry.py` and the startup receipt (§2.3) |
| 0 % hits in chat today | `chat_call_metrics`, SEE-4919/SEE-5076 (the lead's 19 Oct measurements) |

## 1. What serves chat, and where its code lives

Chat is answered by container `tensorfold-qwen36` on `dgx-office-02` (`:8890`, model `qwen3.6`):

```
tensorfold serve …/Qwen3.6-35B-A3B-NVFP4 --port 8890 --name qwen3.6 --parallel 16 --context 81920 \
  --max-tokens 16384 --no-update-check        # TF_KEEP_MANY=8 TF_PREFILL_STEP=2048
```

It is built from fork `machinegenieorg/TensorFold`, branch `q36/prod-candidate-0363-pf` @ `c09b5d4`: upstream
0.3.6.3 plus our PRs #80/#83/#84 (the appliance recipe is `appliance/boxes/dgx-office/tensorfold/README.md` in
the seesa repo). **That commit is in no branch of this repository** — here `main` is the Zig engine (1.0.7) and
`python-0.6` is 0.6.6, a later Python line whose `cuda/markers.py` differs from the running file only by one
import, but whose `families/qwen3_5_moe/cuda/engine.py` and `cuda/streams.py` have moved on (§4e). All paths
below are the deployed build's, read in the container at `/opt/tensorfold-c09b5d4/src/tensorfold`.

`--parallel 16` puts the engine on its concurrent path (`families/qwen3_5_moe/cuda/engine.py:73`,
`MultiDecoder`), and that is the path everything here describes. The single-stream path (same file, `generate`)
is exercised only with `--parallel 1`.

## 2. The policy

### 2.1 Where a resume point is taken

`Qwen36Engine.__init__` builds the points once (`engine.py:107`): `resume_points(model_dir)` →
`cuda/markers.py:49` → `snapshot_points` (`:28`). The points are derived **from the prompt's tokens alone**, so
two prompts that agree up to a point compute the same point:

* `starts[1]` — the **second message's start** (`markers.py:39`), i.e. immediately after the first message,
  which is also after whatever the template renders between the two (the tool list), so this point covers
  rules + tools;
* `found[-1]` — the **last assistant message start** (`markers.py:39`), which for a served prompt is the
  generation header the template appends, so it is effectively the prompt's end;
* `MIN_GAP = 256` (`markers.py:10`) — a point within 256 tokens of the prompt start or of the previous point is
  dropped.

`MultiDecoder` copies a state at each point its prefill reaches and at the prompt end unless a point sits within
`MIN_GAP` of it (`multi.py:92,129,133`, `_keep` at `:115`). So **a chat request `[system, user]` keeps exactly
two states**: one at the user header (end of the stable block), one at the assistant header.

There is no token grid, no byte offset, no marker flag and no caller-chosen position: **message starts and the
prompt end are the only resume points**. (The MLX path does have a grid — `engine/prefill_plan.py`,
`step`/`min_chunk` — but the CUDA engine uses that planner for `points()` only.)

### 2.2 Admission and eviction

`cuda/streams.py:76` `PrefixCache`, `keep` entries:

* **Admission**: every kept state is inserted unconditionally (`add`, `:108`) — no size test, no byte test, no
  priority. Identical token ids replace the existing entry, so a block shared by every conversation is **one**
  entry, not one per conversation.
* **Hit**: `longest` (`:84`) takes the longest stored id list that is a **strict prefix** of the prompt (at
  least one prompt token is always left to prefill). A hit makes the entry newest and marks it hot (`_touch`,
  `:100`).
* **Eviction**: over `keep`, the oldest **never-reused** entry goes first, otherwise the oldest (`:113`). A
  block that has been reused is therefore only evicted once every entry is hot.

Two consequences of the strict-prefix rule: a stored state **cannot be truncated** — Qwen3.6's linear layers
carry a recurrent (DeltaNet) state that is a running summary, so a state is reusable only at exactly its own
length — and a state saved from one prompt is useless to any prompt that differs *before* the state's end.

### 2.3 Pool size and what it costs

`keep = KEEP_MANY = TF_KEEP_MANY`, **8 on this box**, default 3 (`engine.py:14`). It is read at import and sizes
the startup geometry (`stream_geometry`, `engine.py:86`), so it cannot be changed without a restart.
The startup receipt on the box:

```
[tensorfold] CUDA rank 0 startup estimate 72.89 GiB within 81.11 GiB; native 262144, allocated prompt/reply
window 81920, cache slots 81924
[tensorfold] 16 streams of 81920 prompt/reply tokens: together, rounds run eagerly; alone, in CUDA graphs
```

(three starts in the container's log had budgets of 79.57 / 81.11 / 81.29 GiB). The geometry reserves
`(streams + keep + 1)` full-attention caches of one window each plus `(2*streams + keep + 2)` DeltaNet states.
Computed for this checkpoint (10 full-attention of 40 layers, 2 KV heads, head_dim 256) at 81,924 slots:

* one context window of KV = `10 × 81,924 × 2 × 256 × 2 × 2` B = **1.56 GiB**;
* one DeltaNet state copy = `30 × (32·128·128·4 + 3·(2·16·128 + 32·128)·2)` B = **61.4 MiB**;
* so **+1 to `keep` ≈ 1.62 GiB** of reservation, and the pool of 8 ≈ 13.0 GiB.

Eight states is ~4 conversation turns (two states each) for the whole fleet. The shared chat block, once shaped
as in §4a, is a single entry that every turn and every user resumes from — the cheapest possible thing to keep.

### 2.4 Knobs

| Knob | Where | Effect |
| --- | --- | --- |
| `TF_KEEP_MANY` | env, `engine.py:14` | pool size in states (default 3; box 8). Startup geometry; restart to change |
| `TF_PREFILL_STEP` | env, `multi.py:22` | rows per prefill step while other streams decode (default 1024; box 2048). Not retention |
| `MIN_GAP` | source `markers.py:10` | 256 tokens: the closest two resume points may be |
| `--context` | flag | per-state cost scales with it: 40960 halves the price of a `keep` |
| `--parallel` | flag | more streams = fewer bytes left for `keep` |

**Five flags that look relevant and do nothing here**: `--prompt-cache-gib`, `--checkpoint-slots`,
`--spill-gib`, `--snapshot-dir` and `--max-snapshots` (`cli.py:74-84`) belong to the MLX/Apple path. `_serve_cuda`
(`cli.py:335`) never forwards them, so on this engine there is **no byte budget for the pool and no disk
persistence at all** — the pinned system blocks, conversation spill and snapshot reload of
`server/checkpoints.py` do not exist on the box. A restart (three in the container's log) starts cold.

## 3. Why a partial system message reuses nothing (measured)

Seventeen greedy `max_tokens: 1` requests in three runs: the reuse phase (first with an inline equivalent of
`tools/probe_prefix_reuse.py`, then with the committed script), and the churn phase between them. Run 1 and run 2:

```
-- reuse: whole message vs shared leading block
A, user one (cold)                prompt_tokens=   1288 cached_tokens=      0 wall=    223 ms
A, user two (same system)         prompt_tokens=   1288 cached_tokens=   1276 wall=     38 ms
B, user three (B shares its head) prompt_tokens=   1297 cached_tokens=      0 wall=    209 ms

-- churn: does a twice-used block survive distinct prompts?
A before (touch the block)        prompt_tokens=   1287 cached_tokens=   1276 wall=     53 ms
cold 0                            prompt_tokens=    322 cached_tokens=      0 wall=    119 ms
… cold 1–7 identical …
A after 8 cold prompts            prompt_tokens=   1287 cached_tokens=   1276 wall=     35 ms
block still resident
```

* Same first message, different user text: **1,276 of 1,288 tokens reused**, 223 → 38 ms.
* A second first message that shares its **whole leading block and differs after it**: **0 of 1,297 tokens
  reused**, 209 ms. That is the lead's 19 Oct finding, reproduced.
* The twice-used block still hit after eight distinct single-use prompts had turned the 8-state pool over twice
  (each cold request keeps up to two states). Repeated four minutes later, after the churn, the reuse phase
  returned the same `1276 / 1276 / 0` at 52 / 35 / 206 ms: the block was still there, and only the block.

The mechanism, in order: (a) a state is reusable only where one was taken, and none is taken inside a message;
(b) the shared head of two *different* system messages is therefore never a point; (c) a state cannot be
truncated to the shared head afterwards. Chat puts the per-turn sources inside the system message
(SEE-5076), so every turn's first message differs before the only available point, and the 0 % in
`chat_call_metrics` is exactly what this policy predicts. Note also that the pool is not the problem in that
case: 8 states that never match anything are worth nothing.

## 4. Recommended settings — no engine change needed

**a. Make the stable block a whole message.** The point at the second message's start already *is* the end of
the first message, so a byte-identical first message (the rules) with a byte-identical tool list and everything
per-turn in later messages gives a resume point exactly at the end of the stable block. That is
`CHAT_CACHEABLE_PROMPT` (seesa #895): it measured 3,842 of 3,855 tokens reused through the box's proxy, and §3
reproduces the mechanism on a synthetic block of the same shape — 1,276 of 1,288 tokens, 223 → 38 ms on a
1.3k prompt; at chat's ~13k prompt the journey map's step 4 expects ~1.6 s → ~0.1–0.3 s.
No engine change, no new flag on this side.

**b. Keep the tool list byte-identical too.** The point sits *after* the tools: #895 reused 3,842 of 3,855
with a system message of rules alone, so the whole pre-user stretch (rules and tools) is on the reusable side.
SEE-4919 counted 4 distinct tool sets in 15 calls and 10 distinct "static" prefixes; each one moves the point
and misses. A stable *leading* block inside a varying first message is not a point at all today (§5).

**c. `TF_KEEP_MANY` 8 → 10.** Two more states ≈ 3.2 GiB, which fits every budget the box has printed
(≥79.57 GiB against the 72.89 GiB estimate); 12 only if a start prints ≥80.5 GiB, since 12 costs 6.5 GiB more.
Do not set it blind: `--context 81920` is explicit, and if the geometry no longer fits, `capacity.choose`
refuses to start rather than shrinking. The bigger pool is the only lever for the churn case (§3's `cold 0–8`);
what it buys is more distinct conversation prefixes and more slack before an unused block is evicted.

**d. A keep-warm request after a start, if nobody is asking.** A restart drops the block; one tiny shaped
request per restart (rules + tools + one word) puts it back for ~0.4 s at an idle moment. `qwen36-warmup` is
retired and was for JIT, not for this (the appliance's qwen36 README), so this would be new, and optional: the
first real turn pays the same 0.4 s once.

**e. If a rebuild comes from this repository's Python line, keep the knob.** `origin/python-0.6` (0.6.6) has
`KEEP_MANY = 3` **hardcoded** and gains memory-pressure eviction (`cuda/streams.py` `KVRoom`, evicting kept
states by bytes). Rebuilt from 0.6.6 as it stands, the box's pool silently drops 8 → 3. Port the `TF_KEEP_MANY`
read, or set the constant to 8. The `KVRoom` idea is the right direction for §3's churn case — many small
shared blocks stay, long conversation ends go — and is a GB10 measurement, not a decision to take on faith.

## 5. If a sub-message boundary must be reusable: the minimal patch

Today the only boundary you can choose is a message boundary. When the stable block has to be part of a larger
first message, a caller-named marker makes it a resume point: `cuda/markers.py` gains an optional marker
sequence, tokenised from `TF_RESUME_MARKER` in `resume_points` and added to `wanted` next to `starts[1]` and
`found[-1]` (~12 lines, subject to `MIN_GAP`, no change to `engine/prefill_plan.py` or the MLX path, behaviour
unchanged when the variable is unset). The two properties that make it safe are already true of the planner: positions come from
tokens alone, so two prompts sharing a prefix compute the same point, and the state ends exactly at the marker,
before anything that differs.

Cost: one more entry per request out of `keep`, so half as many conversations retained (≈1.62 GiB a slot at
81920, §2.3). A token grid instead of a marker does not work: fine enough to catch a 1.6k shared head
(`MIN_GAP = 256`) it keeps five states for a 1.3k prompt and empties the pool. **Not built**: for journey step 4
the boundary is a message boundary and §4a needs no code, so this stays a proposal until a caller wants a
non-message boundary.

## 6. Test plan on GB10

1. **Shape, then reuse.** On the box, send the shaped prompt (`system` = rules only, `tools` as today, per-turn
   block in the user turn) twice with different user text: pass = `cached_tokens ≈ prompt_tokens − len(user
   content)`, cold wall time ≈ 5–6× the hit (§3's 223 → 38 ms on 1.3k; expect ~1.6 s → ~0.1–0.3 s at 11–13k).
   `tools/probe_prefix_reuse.py --phase reuse` is the shape of the request.
2. **The fleet share.** Two *different* conversations with the same rules+tools: the second must hit the first's
   block. This is the claim the epic leans on; it is the same entry (§2.2).
3. **Retention under load.** 16 concurrent conversations × 3 turns, then the block again; pass = every turn
   after the first reports `cached_tokens ≥ block size`, and no turn between them returns 0 having shared the
   block. This is the case §2.3's pool size governs.
4. **The knob.** Only if `keep` is raised: the start line must still print `allocated prompt/reply window 81920`
   and a startup estimate inside the budget; a refusal to start means it does not fit, revert.
5. **Exactness.** Per CONTRIBUTING, a resumed path must equal a fresh execution: same shaped prompt with
   `"draft": false` (fresh prefill, no reuse) and normally, compare `token_sha`. Reuse must not change output.
6. **Instrumentation.** `chat_call_metrics.cached_tokens` per call (SEE-5048) is the field the scoreboard
   (SEE-5108) needs; the probes here are the before/after, the per-call rows are the steady state.

## 7. Not done

* No box, container or service change: `TF_KEEP_MANY` is still 8 and `--context` still 81920.
* No engine patch or branch: §4 shows the win comes from prompt shape, and the running commit is in no branch
  of this repository, so a patch here could not be built or verified against the box.
* No quality comparison of the reshaped prompt (SEE-5108 owns that; sources leaving the system role changes
  what the model sees).
* Retention was probed with 1.3k prompts and one churn, not with chat's 13k prompts under 16 concurrent turns
  (§6.3 is that measurement).
