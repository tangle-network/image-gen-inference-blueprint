use serde::{Deserialize, Serialize};
use blueprint_std::fmt;
use blueprint_std::path::PathBuf;

use crate::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration
    pub tangle: TangleConfig,

    /// Diffusion backend configuration
    pub diffusion: DiffusionConfig,

    /// HTTP server configuration
    pub server: ServerConfig,

    /// Billing / ShieldedCredits configuration
    pub billing: BillingConfig,

    /// GPU configuration
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional)
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

impl fmt::Debug for OperatorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OperatorConfig")
            .field("tangle", &self.tangle)
            .field("diffusion", &self.diffusion)
            .field("server", &self.server)
            .field("billing", &self.billing)
            .field("gpu", &self.gpu)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TangleConfig {
    /// JSON-RPC endpoint for the Tangle EVM chain
    pub rpc_url: String,

    /// Chain ID
    pub chain_id: u64,

    /// Operator's private key (hex, without 0x prefix)
    pub operator_key: String,

    /// Tangle core contract address
    pub tangle_core: String,

    /// ShieldedCredits contract address
    pub shielded_credits: String,

    /// Blueprint ID this operator is registered for
    pub blueprint_id: u64,

    /// Service ID (set after service activation)
    pub service_id: Option<u64>,
}

impl fmt::Debug for TangleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TangleConfig")
            .field("rpc_url", &self.rpc_url)
            .field("chain_id", &self.chain_id)
            .field("operator_key", &"[REDACTED]")
            .field("tangle_core", &self.tangle_core)
            .field("shielded_credits", &self.shielded_credits)
            .field("blueprint_id", &self.blueprint_id)
            .field("service_id", &self.service_id)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionConfig {
    /// Model identifier (e.g. "stabilityai/stable-diffusion-xl-base-1.0", "black-forest-labs/FLUX.1-dev")
    pub model: String,

    /// Diffusion backend endpoint (ComfyUI, A1111, or diffusers HTTP server).
    /// Overridable via DIFFUSION_ENDPOINT env var.
    #[serde(default = "default_diffusion_endpoint")]
    pub endpoint: String,

    /// Default number of inference steps
    #[serde(default = "default_steps")]
    pub default_steps: u32,

    /// Default image width
    #[serde(default = "default_width")]
    pub default_width: u32,

    /// Default image height
    #[serde(default = "default_height")]
    pub default_height: u32,

    /// Supported output resolutions (WxH strings, e.g. "1024x1024")
    #[serde(default = "default_supported_resolutions")]
    pub supported_resolutions: Vec<String>,

    /// Request timeout for a single generation call (seconds)
    #[serde(default = "default_generation_timeout")]
    pub generation_timeout_secs: u64,

    /// Maximum images per request
    #[serde(default = "default_max_images")]
    pub max_images: u32,

    /// Operations this backend supports: "generate", "edit", "variation", "upscale"
    #[serde(default = "default_supported_operations")]
    pub supported_operations: Vec<String>,

    /// Maximum upload image size in bytes (for edit/variation endpoints)
    #[serde(default = "default_max_image_size_bytes")]
    pub max_image_size_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,

    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,

    /// Maximum request body size in bytes (default 2 MiB)
    #[serde(default = "default_max_request_body_bytes")]
    pub max_request_body_bytes: usize,

    /// Per-request timeout in seconds
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,

    /// Maximum concurrent requests per credit account.
    /// 0 = unlimited (default).
    #[serde(default)]
    pub max_per_account_requests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    /// Whether billing is required for HTTP requests
    #[serde(default = "default_billing_required")]
    pub required: bool,

    /// Price per image in tsUSD base units (6 decimals: 1 = 0.000001 tsUSD)
    pub price_per_image: u64,

    /// Price multiplier for higher resolutions (per megapixel above 1MP).
    /// 0 = flat rate regardless of resolution.
    #[serde(default)]
    pub price_per_extra_megapixel: u64,

    /// Maximum amount a single SpendAuth can authorize
    pub max_spend_per_request: u64,

    /// Minimum balance required in a credit account
    pub min_credit_balance: u64,

    /// Whether billing is required on every request
    #[serde(default = "default_billing_required")]
    pub billing_required: bool,

    /// Minimum charge amount per request (gas cost protection)
    #[serde(default)]
    pub min_charge_amount: u64,

    /// Maximum retries for claim_payment on-chain calls
    #[serde(default = "default_claim_max_retries")]
    pub claim_max_retries: u32,

    /// Clock skew tolerance in seconds for SpendAuth expiry checks
    #[serde(default = "default_clock_skew_tolerance")]
    pub clock_skew_tolerance_secs: u64,

    /// Maximum gas price in gwei (0 = no cap)
    #[serde(default)]
    pub max_gas_price_gwei: u64,

    /// Path to persist used nonces across restarts
    #[serde(default = "default_nonce_store_path")]
    pub nonce_store_path: Option<PathBuf>,

    /// ERC-20 token address for x402 payment
    #[serde(default)]
    pub payment_token_address: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Expected number of GPUs
    pub expected_gpu_count: u32,

    /// Minimum required VRAM per GPU in MiB
    pub min_vram_mib: u32,

    /// GPU model name for on-chain registration
    #[serde(default)]
    pub gpu_model: Option<String>,

    /// GPU monitoring interval in seconds
    #[serde(default = "default_monitor_interval")]
    pub monitor_interval_secs: u64,
}

// Defaults

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
    20 * 1024 * 1024 // 20 MiB
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_max_concurrent() -> usize {
    8
}

fn default_billing_required() -> bool {
    true
}

fn default_monitor_interval() -> u64 {
    30
}

fn default_max_request_body_bytes() -> usize {
    2 * 1024 * 1024
}

fn default_request_timeout_secs() -> u64 {
    300
}

fn default_claim_max_retries() -> u32 {
    3
}

fn default_clock_skew_tolerance() -> u64 {
    30
}

fn default_nonce_store_path() -> Option<PathBuf> {
    Some(PathBuf::from("data/nonces.json"))
}

impl OperatorConfig {
    /// Load config from file, env vars, and CLI overrides.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        // Environment variables override file config.
        // Prefix: IMGGEN_OP_ (e.g. IMGGEN_OP_TANGLE__RPC_URL)
        builder = builder.add_source(
            config::Environment::with_prefix("IMGGEN_OP")
                .separator("__")
                .try_parsing(true),
        );

        // Allow DIFFUSION_ENDPOINT env var to override diffusion.endpoint
        if let Ok(endpoint) = std::env::var("DIFFUSION_ENDPOINT") {
            builder = builder.set_override("diffusion.endpoint", endpoint)?;
        }

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }

    /// Calculate cost for a single image at a given resolution.
    pub fn image_cost(&self, width: u32, height: u32) -> u64 {
        let base = self.billing.price_per_image;
        if self.billing.price_per_extra_megapixel == 0 {
            return base;
        }
        let megapixels = (width as u64 * height as u64) as f64 / 1_000_000.0;
        let extra_mp = (megapixels - 1.0).max(0.0);
        base + (extra_mp * self.billing.price_per_extra_megapixel as f64) as u64
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
                "tangle_core": "0x0000000000000000000000000000000000000001",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "diffusion": {
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "endpoint": "http://127.0.0.1:8188"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "billing": {
                "price_per_image": 50000,
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
        assert_eq!(cfg.billing.price_per_image, 50000);
        assert_eq!(cfg.gpu.expected_gpu_count, 1);
    }

    #[test]
    fn test_defaults_applied() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.diffusion.default_steps, 30);
        assert_eq!(cfg.diffusion.default_width, 1024);
        assert_eq!(cfg.diffusion.default_height, 1024);
        assert_eq!(cfg.diffusion.max_images, 4);
        assert_eq!(cfg.server.max_concurrent_requests, 8);
    }

    #[test]
    fn test_image_cost_flat() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        // price_per_extra_megapixel defaults to 0, so flat rate
        assert_eq!(cfg.image_cost(1024, 1024), 50000);
        assert_eq!(cfg.image_cost(512, 512), 50000);
    }
}
