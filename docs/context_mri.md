# Context MRI

> See what your agent is about to know before it knows it.

Context MRI renders a compiled context (via its BOM) into a **self-contained HTML
report** — no remote JavaScript, no network calls, secrets masked. Install the
optional extra for the richest experience (the core renderer needs no extra deps):

```bash
pip install "ctxbudgeter[viz]"
```

```python
from ctxbudgeter.viz import ContextMRI

ContextMRI.from_compiled(compiled).export_html("context_mri.html")
# or from a saved BOM:
ContextMRI.from_bom("context_bom.json").export_html("context_mri.html")
```

CLI:

```bash
ctxbudgeter compile . --task "..." --bom context_bom.json
ctxbudgeter viz context_bom.json --out context_mri.html
```

## Panels

1. **Summary cards** — tokens, included/excluded/redacted, health, risk, cache efficiency.
2. **Context Window Map** — the exact compiled order; block width ∝ tokens. Encodings:
   required = thick border, cacheable = blue outline, redacted = striped/faded,
   high-risk = red marker.
3. **Source → Context Flow** — raw source → transformation (redact/compress/unchanged) → final item.
4. **Risk Heatmap** — items × {pii, secrets, stale, untrusted, policy, oversized, cache_breaking, duplicate}.
5. **Cache Boundary** — stable prefix vs dynamic section, estimated cacheable tokens, warnings.
6. **Context Waste Report** — token-heavy/low-relevance, high-risk/low-value, cache-breaking items.
7. **Influence Proxy Map** — see the disclaimer below.
8. **Recommendations** — concrete actions.

## The Influence Proxy is NOT model attention

> **Context MRI does not visualize model attention.** It visualizes deterministic
> context structure, token allocation, provenance, policy risk, cache layout, and a
> transparent **Influence Proxy**.

The Influence Proxy is computed from metadata you can inspect:

```
influence_proxy = task_similarity * 0.35
                + source_trust    * 0.20
                + priority        * 0.15
                + freshness       * 0.10
                + retrieval_rank  * 0.10
                + compactness     * 0.10
```

It approximates how much an item is *likely* to shape the response — useful for
spotting high-cost/low-value context. It is **not** a claim about the model's
internals. This disclaimer is rendered inside every HTML report.
