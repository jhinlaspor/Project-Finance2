#!/usr/bin/env bash
set -euo pipefail
echo "→ Codex setup script: installing build tools & Python deps..."

# 1. Ensure build-essentials for any wheels that need compilation
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential

# 2. Use Codex proxy certificate so pip/apt can reach the internet
export PIP_CERT="$CODEX_PROXY_CERT"

# 3. Upgrade pip + install deps
python -m pip install --upgrade pip wheel
python -m pip install --no-cache-dir -r requirements.txt

# 4. Create a minimal .env so tests don’t choke if user forgot to set vars
cat <<EOF > .env
APCA_API_KEY_ID=dummy
APCA_API_SECRET_KEY=dummy
APCA_API_BASE_URL=https://paper-api.alpaca.markets
SIMULATION=true
EOF

echo "✓ Environment ready – handoff to Codex agent"
