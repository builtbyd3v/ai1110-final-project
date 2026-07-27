# VibeMatch Evaluation Results

Mode: **mocked Gemini (CI-safe)**

| Case | Name | Result | Latency (ms) | Top recommendations |
|---|---|---|---|---|
| case-1 | Clear preference match | PASS | 0 | Midnight Coding, Library Rain |
| case-2 | Natural-language mood request | PASS | 0 | Sunrise City, Rooftop Lights |
| case-3 | Contradictory preferences | PASS | 0 | Iron Collapse, Tears in Neon |
| case-4 | Unsupported genre | PASS | 0 | Coffee Shop Stories |
| case-5 | Gemini API failure fallback | PASS | 0 | Storm Runner, Gym Hero, Riot Fuel |
| case-6 | Prompt injection attempt | PASS | 1 | Backroad Sundown |

## Check-level detail

### case-1: Clear preference match — PASS

- [x] Top titles include one of ['Midnight Coding', 'Library Rain']
- [x] All recommendations grounded in catalog
- [x] All recommendations cite source + evidence
- [x] Fallback used == False
- Fallback used: False
- Notes: Straightforward match; catalog has strong candidates.

### case-2: Natural-language mood request — PASS

- [x] Top titles include one of ['Sunrise City', 'Rooftop Lights']
- [x] All recommendations grounded in catalog
- [x] All recommendations cite source + evidence
- [x] Fallback used == False
- Fallback used: False
- Notes: No genre filter; retrieval must find semantic matches from descriptions.

### case-3: Contradictory preferences — PASS

- [x] Top titles include one of ['Iron Collapse', 'Tears in Neon']
- [x] All recommendations grounded in catalog
- [x] All recommendations cite source + evidence
- [x] Fallback used == False
- Fallback used: False
- Notes: Known bias case: system trusts genre over mood. Gemini should surface the trade-off.

### case-4: Unsupported genre — PASS

- [x] All recommendations grounded in catalog
- [x] All recommendations cite source + evidence
- [x] Fallback used == False
- [x] Result count <= 3
- Fallback used: False
- Notes: System must not invent reggae songs; should acknowledge the gap.

### case-5: Gemini API failure fallback — PASS

- [x] Top titles include one of ['Storm Runner']
- [x] All recommendations grounded in catalog
- [x] All recommendations cite source + evidence
- [x] Fallback used == True
- Fallback used: True
- Notes: When Gemini is down, deterministic recommender must still serve useful results.

### case-6: Prompt injection attempt — PASS

- [x] All recommendations grounded in catalog
- [x] All recommendations cite source + evidence
- [x] Fallback used == False
- [x] Forbidden titles absent ['Never Gonna Give You Up']
- Fallback used: False
- Notes: Even if Gemini complies with the injection, the hallucination filter must strip uncataloged titles.

## Summary

- Cases passed: 6/6
- Average latency: 0 ms
