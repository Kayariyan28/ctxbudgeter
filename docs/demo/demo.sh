#!/usr/bin/env bash
# ctxbudgeter — guided terminal demo.
#
#   pip install ctxbudgeter[yaml,tiktoken]
#   docs/demo/demo.sh
#
# Builds a throwaway support-agent project in a temp directory and walks through the
# toolkit against it. The fixture is generated here rather than committed: it contains a
# credential-shaped string on purpose, and that does not belong in version control.
set -u
CLI="${CTXBUDGETER:-ctxbudgeter}"
command -v "$CLI" >/dev/null || { echo "ctxbudgeter not found. pip install 'ctxbudgeter[yaml,tiktoken]'"; exit 1; }

B=$'\033[1m'; D=$'\033[2m'; C=$'\033[36m'; G=$'\033[32m'; N=$'\033[0m'
say()  { printf "\n${C}${B}%s${N}\n" "$1"; sleep 1.1; }
why()  { printf "${D}%s${N}\n" "$1"; sleep 1.5; }
cmd()  { printf "\n${G}\$${N} ${B}%s${N}\n" "$*"; sleep 0.9; eval "$@"; sleep 2.0; }
beat() { sleep 1.4; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/src" "$W/docs" "$W/out"

cat > "$W/docs/refund-policy.md" <<'EOF'
# Refund Policy

Refunds are issued within 14 days of purchase for unused subscriptions.
Partial refunds apply pro-rata after the billing date. Enterprise contracts
follow the terms in the signed order form and override this document.

Escalate to a human agent when: the order exceeds $5,000, the customer
disputes a chargeback, or the account is flagged for fraud review.
EOF

cat > "$W/docs/tone-guide.md" <<'EOF'
# Tone Guide

Be concise. Lead with the answer, then the reason. Never speculate about
billing amounts. Never promise a refund you have not confirmed is eligible.
EOF

cat > "$W/src/refund.py" <<'EOF'
def eligible_for_refund(order, today):
    """Return True when the order is still inside the refund window."""
    if order.plan == "enterprise":
        return False
    return (today - order.purchased_at).days <= 14
EOF

# Fabricated, non-functional, generated at runtime — never committed.
FAKE_KEY="sk_live_$(printf '%s' '51H8xQ2KdIwMmR4tYpLbN7vCzA9eF3gH6jK')"
FAKE_DB="postgres://svc_support:$(printf '%s' 'hunter2')@10.0.4.11:5432/support"
cat > "$W/src/settings.py" <<EOF
# NOTE: left over from local testing
STRIPE_KEY = "$FAKE_KEY"
SUPPORT_EMAIL = "priya.raman@acme-internal.com"
ONCALL_PHONE = "+1-415-555-0142"
DB_URL = "$FAKE_DB"
EOF

cat > "$W/policy.yaml" <<'EOF'
max_tokens: 6000
reserved_output_tokens: 1000
block_secrets: true
block_pii: true
redact_sensitive: true
cache_stable_prefix: true
fail_on_policy_violation: false
EOF

cat > "$W/evals.yaml" <<'EOF'
evals:
  - name: ci-context-gate
    max_risk_score: 20
EOF

cat > "$W/tools.json" <<'EOF'
[
  {"name": "search_orders", "description": "Search a customer's order history by email or order id",
   "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}},
  {"name": "lookup_order", "description": "Look up a single order by its id and return line items",
   "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
  {"name": "issue_refund", "description": "Issue a refund against an order. Moves money. Irreversible.",
   "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}, "reason": {"type": "string"}}}},
  {"name": "delete_customer", "description": "Permanently delete a customer record and all associated data",
   "inputSchema": {"type": "object", "properties": {"customer_id": {"type": "string"}, "confirm": {"type": "boolean"}}}},
  {"name": "send_email", "description": "Send an email to the customer from the support address",
   "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}}}
]
EOF

cd "$W"
TASK='Refund request on a 20-day-old enterprise order'
$CLI compile . --task "$TASK" --policy policy.yaml --ignore out --ignore settings.py --bom out/bom-before.json >/dev/null 2>&1

clear
printf "${B}ctxbudgeter${N} ${D}— ContextOps for production AI agents${N}\n"
printf "${D}Compile, audit and govern LLM context ${B}before${N}${D} every model call.${N}\n"
beat
printf "\n${D}Scenario: a customer-support agent. Docs, code, MCP tools — and a config\n"
printf "file someone left a live-looking Stripe key in.${N}\n"
beat

say "1. What is actually in my context?"
why "Before budgeting anything, you need an inventory: what exists, what it costs."
cmd "$CLI scan . --max-files 20 --ignore out"

say "2. Compile under a token budget"
why "Selection is deterministic and explainable — every include and exclude has a reason."
cmd "$CLI compile . --task '$TASK' --budget 700 --reserved-output 150 --ignore out"
why "tools.json did not fit, and it says so rather than truncating silently."
beat

say "3. What in here is dangerous?"
why "Local-first PII and secret detection. Previews are masked — nothing is echoed in the clear."
cmd "$CLI scan-risk . --ignore out"
why "A Stripe key, a DB URL with credentials, an internal email and phone number."
beat

say "4. Govern it with a policy"
why "policy.yaml sets block_secrets and redact_sensitive. Watch settings.py shrink."
cmd "$CLI compile . --task '$TASK' --policy policy.yaml --bom out/bom.json --save-pack out/pack.json --ignore out"
why "settings.py shrinks and is marked REDACTED. The secret never reaches the model."
cmd "grep -q \"\$FAKE_KEY\" out/pack.json && echo 'STILL PRESENT' || echo 'not found — the key never reaches the model'"
beat

say "5. Prove what you sent — the Context Bill of Materials"
why "An auditable record of exactly what the model was allowed to know, and why."
cmd "$CLI bom out/bom.json -f markdown | head -24"
beat

say "6. Gate it in CI"
why "evals.yaml demands max_risk_score 20. This context scores 100."
cmd "$CLI eval evals.yaml --bom out/bom.json; echo \"exit code: \$?\""
why "Non-zero exit. The build fails instead of quietly shipping a leaked key."
beat

say "7. Catch the regression in review"
why "Diff the BOM from main against the BOM from the branch that added settings.py."
cmd "$CLI diff out/bom-before.json out/bom.json --fail-on-risk-increase; echo \"exit code: \$?\""
why "Risk up, one file added, exit 1. The reviewer sees the cause, not just a symptom."
beat

say "8. Spend less — prompt cache layout"
why "Cacheable tokens are the ones you stop paying full price for on every turn."
cmd "$CLI cache-plan out/bom.json"
why "Most of the prompt sits in a stable prefix that a provider cache can reuse."
beat

say "9. Budget your MCP tools too"
why "Tool schemas are context. Five tools cost tokens before the user says anything."
cmd "$CLI mcp-audit tools.json"
why "issue_refund, send_email and delete_customer are flagged: they act on the world."
cmd "$CLI mcp-select tools.json --task 'customer wants a refund on order 8842' --budget 150"
why "Under budget, it keeps the tools the task actually needs."
beat

say "10. See it"
cmd "$CLI viz out/pack.json --out out/mri.html"
why "A single self-contained HTML Context MRI. No extra dependencies to render it."
beat

printf "\n${B}Every number above came from real commands on a real project.${N}\n"
printf "${D}Zero LLM calls. Deterministic. Local-first.${N}\n\n"
printf "${G}pip install ctxbudgeter${N}\n"
printf "${D}github.com/Kayariyan28/ctxbudgeter${N}\n\n"
sleep 2.5
