# AI Interactions Log

> **Stretch features only.** Only fill in the sections that apply to stretch features you attempted. If you did not attempt a stretch feature, leave its section blank or delete it. This file is not required for the core project.

---

## Agentic Workflow (SF8)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

"Add 5 or more complex attributes to the dataset that aren't currently
present (like popularity, release decade, or detailed mood tags), update
both `data/songs.csv` and the scoring logic in `src/recommender.py` so
scoring accounts for the new attributes, and don't break the existing
tests or the default CLI output." Same session also covered the other
three bonus challenges (scoring modes, diversity penalty, table output)
as one continuous multi-file task.

**Prompts used:**

- "Add 5 new columns to songs.csv (popularity 0-100, release_decade,
  detailed_mood_tags, instrumentalness, language), fill in plausible
  values for all 18 existing rows, then wire optional scoring bonuses
  for each into `score_song()` and `Recommender._score()` that only
  activate when the user profile actually specifies that preference."
- "Don't make the new Song/UserProfile fields required, existing tests
  construct both with only the original fields."
- "Add a `mode` argument (`balanced`/`genre-first`/`mood-first`/
  `energy-focused`) to the scoring functions using a Strategy-pattern
  style dict of weight presets, switchable from the CLI as
  `python -m src.main <mode>`."
- "Add a diversity penalty to `recommend_songs()` only (not the OOP
  `Recommender`, since the test fixture reuses one artist across two
  songs and expects both back): greedily re-rank, penalizing a
  remaining candidate's score once its artist or a triple-repeated
  genre is already in the picked list."
- "Replace the per-line CLI print with a small stdlib-only ASCII table
  (no new dependency), Title / Score / Reasons columns."

**What did the agent generate or change?**

- `data/songs.csv`: added 5 columns across all 18 rows
- `src/recommender.py`: `Song`/`UserProfile` new optional fields,
  `_score_attributes()` extended with weight presets and the 5 new
  scoring dimensions, `SCORING_MODES` dict, diversity-penalty logic
  added inside `recommend_songs()`
- `src/main.py`: CLI mode argument, `_format_table()` helper
- Ran `pytest` (2 passed) and `python -m src.main` (plus all 4 modes,
  plus a manual profile using every new attribute) to check real output
  before writing anything into the docs

Advanced-profile example run (`lofi/chill/0.4` plus popularity,
decade, mood tags, instrumentalness, and language preferences), showing
all 5 new scoring dimensions firing at once:

```
Midnight Coding - Score: 6.92
Because: genre match (+2.0), mood match (+1.0), energy similarity (+0.98), popularity similarity (+0.46), release decade match (+0.5), mood tag overlap x2 (+1.00), instrumentalness similarity (+0.47), language match (+0.5)
Library Rain - Score: 6.42
Because: genre match (+2.0), mood match (+1.0), energy similarity (+0.95), popularity similarity (+0.47), release decade match (+0.5), mood tag overlap x1 (+0.50), instrumentalness similarity (+0.50), language match (+0.5)
Spacewalk Thoughts - Score: 3.77
Because: mood match (+1.0), energy similarity (+0.88), popularity similarity (+0.42), release decade match (+0.5), instrumentalness similarity (+0.48), language match (+0.5)
Focus Flow - Score: 2.70
Because: genre match (+2.0), energy similarity (+1.00), popularity similarity (+0.46), release decade match (+0.5), instrumentalness similarity (+0.49), language match (+0.5), same-artist diversity penalty (-1.50), same-genre diversity penalty (-0.75)
Coffee Shop Stories - Score: 2.62
Because: energy similarity (+0.97), popularity similarity (+0.45), mood tag overlap x2 (+1.00), instrumentalness similarity (+0.20)
```

**What did you verify or fix manually?**

- Caught that the agent's first draft made the 5 new `Song` fields
  required (no defaults), which crashed `tests/test_recommender.py`
  since that fixture never sets them. Fixed by giving them neutral
  defaults (`popularity=50.0`, `release_decade="unknown"`, etc.).
- Caught that adding the diversity penalty to the shared `_score()`
  path would have changed `Recommender.recommend()` too and broken the
  `k=2` test (its two-song fixture shares one artist). Confirmed the
  penalty only belongs in the functional `recommend_songs()` path and
  left the OOP path untouched.
- Reran every profile documented in the README and `model_card.md`
  after the diversity penalty landed, since it changed real output:
  the default CLI sample and the "Chill Lofi" stress-test block both
  had one song swap out. Updated the docs to match instead of leaving
  stale output.

---

## Design Pattern (SF10)

> Document how AI helped you choose or implement a design pattern.

**Which design pattern did you use?**

Strategy, applied to the scoring weights (Bonus Challenge 2: multiple
scoring modes).

**How did AI help you brainstorm or implement it?**

Asked: "I want 3-4 interchangeable ranking strategies (genre-first,
mood-first, energy-focused) selectable at call time, without a big
if/elif chain in the scoring function. What's a clean way to do this in
Python without over-building it?" The answer was a textbook Strategy
pattern, but scaled down: instead of a full class hierarchy (a
`ScoringStrategy` abstract base class plus one concrete subclass per
mode), each "strategy" is just a named dict of weights
(`{"genre": 2.0, "mood": 1.0, "energy": 1.0}`), stored in one
`SCORING_MODES` dict and selected by string key. That's still the
Strategy pattern in spirit (an interchangeable algorithm-selector
looked up at runtime, callers don't know or care which one they got),
just without a class per strategy, since each strategy here is 3
numbers, not behavior, a full class hierarchy would have been
over-engineering for what it does.

**How does the pattern appear in your final code?**

`SCORING_MODES` in `src/recommender.py` (the strategy registry).
`score_song()`, `Recommender._score()`, and `recommend_songs()` all
accept a `mode`/`weights` argument and do
`SCORING_MODES.get(mode, BALANCED_WEIGHTS)` to select a strategy, then
pass it into the shared `_score_attributes()` function, which applies
whichever weights it was handed without knowing which mode it's in.
`src/main.py` exposes the switch as a CLI argument:
`python -m src.main mood-first`.

---

## Applied AI System Upgrade (Final Project)

> The whole final project was built agentically in one extended Cursor
> session. This section documents that workflow per the same stretch rules.

**What task did you give the agent?**

"Take this rule-based CLI recommender and upgrade it into the AI110 final
'Applied AI System': Gemini + RAG, all five optional stretch features
(advanced RAG, multimodal, complex tool use, evaluation framework, polished
UI), keep all four original bonus features working, TDD throughout, and
don't commit without asking."

**How the work was decomposed:**

- Part 0: environment (deps, `.env` template, gitignore, CI, README skeleton)
- Part 1: catalog enrichment (`description`/`listening_context` on all 18
  songs) + consolidating diversity logic into one shared path for both
  the OOP and functional recommenders
- Part 2: hybrid retrieval (`src/rag.py`) — keyword, Gemini embeddings,
  deterministic re-rank, rank fusion, query expansion, embedding cache
- Part 3: tools (`src/tools.py`), multimodal (`src/multimodal.py`),
  orchestration (`src/ai_service.py`) with grounded prompts,
  hallucination filter, and deterministic fallback
- Part 4: Streamlit UI (`src/app.py`)
- Part 5: evaluation harness (`evaluation/`) + edge-case test growth
- Stretch completion: Gemini function-calling tool loop, iTunes
  artwork/preview enrichment, live cache build
- Part 6: documentation (this file, README, model card)

**What did you verify or fix manually?**

- **Deprecated model caught by a live smoke test.** The first default
  (`gemini-2.5-flash`) returned a 404 "no longer available to new users"
  only when run against the real API — mocked tests could never have
  caught it. Migrated to `gemini-3.5-flash` (GA) after checking Google's
  deprecation table.
- **Hallucination filter scope bug caught by the evaluation harness.**
  The filter checked titles against *retrieved* documents, not the full
  catalog, so a legitimate catalog song was silently dropped whenever
  retrieval hadn't surfaced it. Case-6 ("prompt injection") exposed it
  by returning an empty list. Fixed to filter against the full catalog +
  live result set; the case now strips the injected song and keeps the
  real one.
- **Tool-evidence merge bug caught by a unit test.** Tool-called iTunes
  results never reached the hallucination filter, so every tool-discovered
  track was filtered out. Fixed by accumulating tool outputs into the
  evidence list.
- **Live quota outage became a feature demo.** Free-tier Gemini quota
  (20 req/day) ran out mid-verification; the app served deterministic
  recommendations with a visible warning instead of crashing — exactly
  the designed fallback behavior, confirmed working under real
  conditions rather than a mock.
- **Evaluation run twice:** mocked (CI-safe, `results.md`) and live
  against the real API (`results-live.md`), 6/6 both modes.
