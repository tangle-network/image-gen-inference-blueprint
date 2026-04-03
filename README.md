![Tangle Network Banner](https://raw.githubusercontent.com/tangle-network/tangle/refs/heads/main/assets/Tangle%20%20Banner.png)

<h1 align="center">Image Generation Blueprint</h1>

<p align="center"><em>Decentralized image generation on <a href="https://tangle.tools">Tangle</a> — operators serve Stable Diffusion, FLUX, and SDXL via ComfyUI or diffusers.</em></p>

<p align="center">
  <a href="https://discord.com/invite/cv8EfJu3Tn"><img src="https://img.shields.io/discord/833784453251596298?label=Discord" alt="Discord"></a>
  <a href="https://t.me/tanglenet"><img src="https://img.shields.io/endpoint?color=neon&url=https%3A%2F%2Ftg.sumanjay.workers.dev%2Ftanglenet" alt="Telegram"></a>
</p>

## Overview

A Tangle Blueprint enabling operators to serve image generation models with anonymous payments through shielded credits. Operators run ComfyUI, A1111, or diffusers-based endpoints and register on-chain with GPU capabilities.

**Dual payment paths:**
- **On-chain jobs** via TangleProducer — verifiable results on Tangle
- **x402 HTTP** — fast private image generation at `/v1/images/generations`

OpenAI Images API compatible. Built with [Blueprint SDK](https://github.com/tangle-network/blueprint) with TEE support.

## Components

| Component | Language | Description |
|-----------|----------|-------------|
| `operator/` | Rust | Operator binary — wraps diffusion endpoint, HTTP server, SpendAuth billing |
| `contracts/` | Solidity | ImageGenBSM — GPU validation, per-image pricing, resolution tracking |

## Pricing

Per-image pricing with optional per-megapixel surcharge for high resolutions. Configured by the Blueprint admin via `configureModel()`.

## TEE Support

Add `features = ["tee"]` to `blueprint-sdk` in Cargo.toml. The `TeeLayer` middleware transparently attaches attestation metadata when running in a Confidential VM (H100 CC, SEV-SNP, TDX). Passes through when no TEE is configured.

## Quick Start

```bash
# Configure
cp config/operator.example.toml config/operator.toml
# Edit: model, GPU specs, diffusion endpoint URL, pricing

# Build
cargo build --release

# Run (requires a running diffusion server)
DIFFUSION_ENDPOINT=http://localhost:7860 ./target/release/image-gen-operator
```

## Related Repos

- [Blueprint SDK](https://github.com/tangle-network/blueprint) — framework for building Blueprints
- [vLLM Inference Blueprint](https://github.com/tangle-network/vllm-inference-blueprint) — text inference
- [Voice Inference Blueprint](https://github.com/tangle-network/voice-inference-blueprint) — TTS/STT
- [Embedding Blueprint](https://github.com/tangle-network/embedding-inference-blueprint) — text embeddings
- [Video Generation Blueprint](https://github.com/tangle-network/video-gen-inference-blueprint) — video generation
