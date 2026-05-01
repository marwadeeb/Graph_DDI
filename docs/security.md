# Security Posture — DDI Checker

Brief audit of known attack surfaces. This is a public read-only decision-support tool:
no authentication, no user accounts, no stored user data.

---

## Attack Surface Summary

| Vector | Risk | Mitigation |
|---|---|---|
| Drug name input (`drug_a`, `drug_b`) | Injection, unexpected input | Resolved through `resolve_drug()` — rejects anything not in the 4,795-drug list |
| Chat message input (`/api/chat`) | Prompt injection into LLM | Structured system prompt with explicit role constraints; capped at 2,000 chars |
| API JSON body | Malformed / oversized payload | `get_json(silent=True)` returns `{}` on parse failure; no crash path |
| Query string params | Path traversal, unexpected types | `int()` cast with fallback; no file paths derived from user input |
| GROQ_API_KEY | Secret exposure | Loaded from `.env` (gitignored); never returned in any API response |
| Dependency CVEs | Transitive vulnerabilities | Auditable with `pip-audit -r requirements.txt` at each dependency update |

---

## What protects us structurally

**Input is bounded at resolution.**
`resolve_drug()` maps every user-supplied string against a fixed set of ~4,800 known drug
names and IDs. Anything not in that set returns `drug_not_found` immediately — no further
processing occurs. This is the most important security property of the system: the attack
surface for the primary endpoint is bounded to a known vocabulary.

**No SQL — no injection.**
All drug lookups are `frozenset` dict operations on an in-memory Python dict. There is no
query language, no ORM, and no database connection to inject into.

**Jinja2 auto-escaping.**
All template variables are HTML-escaped by default in Flask/Jinja2. User-supplied strings
never reach the browser as raw HTML.

**No file I/O from user input.**
No route reads or writes files based on user-supplied path strings. The only file reads at
request time are from fixed, hardcoded paths.

**LLM prompt injection (chat endpoint).**
The user message is inserted into a structured prompt where the system role explicitly
constrains the model to drug-interaction responses. Full output validation (e.g., refusing
replies that contain code or instructions) is a recommended hardening step for production.

**No persistent user data.**
The recent queries tracked in the dashboard are in-memory only and contain no PII — only
resolved drug names (public pharmacological data) and response metadata. They reset on
server restart. This is consistent with the RM3 (Privacy) design decision.

---

## What we don't have (acceptable for this context)

| Missing control | Why acceptable here | What to add for production |
|---|---|---|
| Rate limiting | Demo/academic tool; no sensitive data at risk | `flask-limiter` — e.g. 60 req/min per IP |
| Authentication | All data is public read-only | Not needed unless usage logs or admin actions are added |
| HTTPS enforcement | Handled by HuggingFace Spaces infrastructure | Enforce `Strict-Transport-Security` header if self-hosted |
| Output validation on LLM | Low-risk for this domain | Regex filter on LLM output for non-medical content |
| Dependency pinning | `requirements.txt` uses `>=` ranges | Pin exact versions + run `pip-audit` in CI |

---

## How to run the dependency audit

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

Known CVEs in development-only dependencies (not present in the deployed Docker image)
are excluded from the threat model.

---

## Why this matters as a DDI tool

Every user input is a potential attack vector — but for this system, the most dangerous
realistic attack is not a technical exploit but a **semantic one**: a user submitting a
drug name that resolves to the wrong drug (e.g. a brand name that maps to multiple generics).
This is why brand names and misspellings are rejected rather than fuzzy-matched:
returning wrong interaction data is more dangerous than returning `not_found`.
This is documented in full in [responsible_ml.md](responsible_ml.md) under RM4.
