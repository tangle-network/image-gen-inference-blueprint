#!/usr/bin/env bash
# Register the image-gen inference blueprint on Tangle.
# Release definitions and local Anvil inputs are documented in README.md.

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"
TANGLE_CORE="${TANGLE_CORE:-0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9}"

MODEL="${MODEL:-black-forest-labs/FLUX.1-dev}"
GPU_COUNT="${GPU_COUNT:-1}"
TOTAL_VRAM="${TOTAL_VRAM:-24000}"
GPU_MODEL="${GPU_MODEL:-NVIDIA RTX 4090}"
ENDPOINT="${ENDPOINT:-https://your-operator.example.com}"

echo "=== Image-Gen Inference Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Tangle Core: $TANGLE_CORE"
echo "Payment:     ${PAYMENT_TOKEN:-<default USDC sepolia>}"
echo "Model:       $MODEL"
echo "GPUs:        $GPU_COUNT x $GPU_MODEL ($TOTAL_VRAM MiB)"
echo "Endpoint:    $ENDPOINT"
echo ""

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CORE_DIR="$(cargo metadata --manifest-path "$ROOT_DIR/Cargo.toml" --format-version 1 --locked | python3 -c 'import json,sys,pathlib; print(pathlib.Path(next(p["manifest_path"] for p in json.load(sys.stdin)["packages"] if p["name"] == "tangle-inference-core")).parent)')"
BLUEPRINT_DEFINITION="$(node "$CORE_DIR/scripts/register-blueprint.mjs" "$ROOT_DIR/deploy/definition.json" "$ROOT_DIR")"
export BLUEPRINT_DEFINITION
trap 'rm -f -- "$BLUEPRINT_DEFINITION"' EXIT
cd "$(dirname "$0")/../contracts"

# Deploy BSM AND register the blueprint in one forge-script broadcast.
DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" \
    TANGLE_CORE="${TANGLE_CORE:-}" \
    PAYMENT_TOKEN="${PAYMENT_TOKEN:-}" \
    forge script script/Deploy.s.sol:Deploy \
        --remappings "tangle-inference-core/=$CORE_DIR/contracts/" \
        --rpc-url "$RPC_URL" \
        --broadcast --slow)

echo "$DEPLOY_OUTPUT"

# Extract BSM address + blueprint ID for downstream scripts.
BSM_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_IMAGE_GEN_BSM=0x[0-9a-fA-F]+' | tail -1 | cut -d= -f2)
BLUEPRINT_ID=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_IMAGE_GEN_BLUEPRINT_ID=[0-9]+' | tail -1 | cut -d= -f2)

if [ -z "$BSM_ADDRESS" ] || [ -z "$BLUEPRINT_ID" ]; then
    echo "ERROR: failed to extract addresses from forge output"
    exit 1
fi

echo ""
echo "=== Blueprint registered ==="
echo "Blueprint ID:    $BLUEPRINT_ID"
echo "ImageGenBSM:     $BSM_ADDRESS"
echo ""
echo "Next step (blueprint owner): configure a model price on the BSM."
echo "  cast send $BSM_ADDRESS \\"
echo "    'configureModel(string,uint64,uint32)' '$MODEL' <pricePerImage> <minGpuVramMib> \\"
echo "    --rpc-url $RPC_URL --private-key \$OWNER_KEY"
echo ""

# Operator registration is a separate step (per-operator). Encode the
# registration inputs so the operator can call Tangle.registerOperator with
# the right calldata.
REG_INPUTS=$(cast abi-encode \
    "f(string,uint32,uint32,string,string)" \
    "$MODEL" "$GPU_COUNT" "$TOTAL_VRAM" "$GPU_MODEL" "$ENDPOINT")

echo "Operator registration inputs (use these to register an operator):"
echo "  $REG_INPUTS"
echo ""
