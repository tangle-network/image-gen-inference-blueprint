//! Image-generation-specific operator configuration.
//!
//! Shared infrastructure config (`TangleConfig`, `ServerConfig`, `BillingConfig`,
//! `GpuConfig`) lives in `tangle-inference-core` and is re-exported here for
//! convenience.

use serde::{Deserialize, Serialize};

pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};

use crate::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration (shared).
    pub tangle: TangleConfig,

    /// Diffusion backend + per-image pricing configuration (image-gen-specific).
    pub diffusion: ImageGenConfig,

    /// HTTP server configuration (shared).
    pub server: ServerConfig,

    /// Billing / ShieldedCredits infrastructure configuration (shared).
    pub billing: BillingConfig,

    /// GPU configuration (shared).
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional — disabled by default).
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

/// Diffusion backend + pricing config. This is the only truly image-gen-specific
/// config section — everything else comes from `tangle-inference-core`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenConfig {
    /// Model identifier (e.g. "stabilityai/stable-diffusion-xl-base-1.0",
    /// "black-forest-labs/FLUX.1-dev").
    pub model: String,

    /// Diffusion backend endpoint (ComfyUI, A1111, or diffusers HTTP server).
    /// Overridable via DIFFUSION_ENDPOINT env var.
    #[serde(default = "default_diffusion_endpoint")]
    pub endpoint: String,

    /// Default number of inference steps.
    #[serde(default = "default_steps")]
    pub default_steps: u32,

    /// Default image width.
    #[serde(default = "default_width")]
    pub default_width: u32,

    /// Default image height.
    #[serde(default = "default_height")]
    pub default_height: u32,

    /// Supported output resolutions (WxH strings, e.g. "1024x1024").
    #[serde(default = "default_supported_resolutions")]
    pub supported_resolutions: Vec<String>,

    /// Request timeout for a single generation call (seconds).
    #[serde(default = "default_generation_timeout")]
    pub generation_timeout_secs: u64,

    /// Maximum images per request.
    #[serde(default = "default_max_images")]
    pub max_images: u32,

    /// Operations this backend supports: "generate", "edit", "variation", "upscale".
    #[serde(default = "default_supported_operations")]
    pub supported_operations: Vec<String>,

    /// Maximum upload image size in bytes (for edit/variation endpoints).
    #[serde(default = "default_max_image_size_bytes")]
    pub max_image_size_bytes: usize,

    /// Flat per-image price in tsUSD base units (6 decimals: 1 = 0.000001 tsUSD).
    pub price_per_image: u64,
}

fn default_diffusion_endpoint() -> String {
    "http://127.0.0.1:8188".to_string()
}

fn default_steps() -> u32 {
    30
}

fn default_width() -> u32 {
    1024
}

fn default_height() -> u32 {
    1024
}

fn default_supported_resolutions() -> Vec<String> {
    vec![
        "512x512".to_string(),
        "768x768".to_string(),
        "1024x1024".to_string(),
        "1024x1792".to_string(),
        "1792x1024".to_string(),
    ]
}

fn default_generation_timeout() -> u64 {
    120
}

fn default_max_images() -> u32 {
    4
}

fn default_supported_operations() -> Vec<String> {
    vec!["generate".to_string()]
}

fn default_max_image_size_bytes() -> usize {
    20 * 1024 * 1024
}

impl OperatorConfig {
    /// Load config from file + env vars.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        // Env vars override file config. Prefix: IMGGEN_OP_
        // (e.g. IMGGEN_OP_TANGLE__RPC_URL).
        builder = builder.add_source(
            config::Environment::with_prefix("IMGGEN_OP")
                .separator("__")
                .try_parsing(true),
        );

        // Allow DIFFUSION_ENDPOINT env var to override diffusion.endpoint.
        if let Ok(endpoint) = std::env::var("DIFFUSION_ENDPOINT") {
            builder = builder.set_override("diffusion.endpoint", endpoint)?;
        }

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_config_json() -> &'static str {
        r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "diffusion": {
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "endpoint": "http://127.0.0.1:8188",
                "price_per_image": 50000
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "billing": {
                "max_spend_per_request": 1000000,
                "min_credit_balance": 1000
            },
            "gpu": {
                "expected_gpu_count": 1,
                "min_vram_mib": 8192
            }
        }"#
    }

    #[test]
    fn test_deserialize_full_config() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.diffusion.model, "stabilityai/stable-diffusion-xl-base-1.0");
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.diffusion.price_per_image, 50000);
        assert_eq!(cfg.gpu.expected_gpu_count, 1);
        assert!(cfg.tangle.service_id.is_none());
    }

    #[test]
    fn test_defaults_applied() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.diffusion.default_steps, 30);
        assert_eq!(cfg.diffusion.default_width, 1024);
        assert_eq!(cfg.diffusion.default_height, 1024);
        assert_eq!(cfg.diffusion.max_images, 4);
        assert_eq!(cfg.server.max_concurrent_requests, 64);
        assert_eq!(cfg.gpu.monitor_interval_secs, 30);
    }

    #[test]
    fn test_missing_required_field_fails() {
        let bad = r#"{"tangle": {"rpc_url": "http://localhost:8545"}}"#;
        let result = serde_json::from_str::<OperatorConfig>(bad);
        assert!(result.is_err());
    }
}
