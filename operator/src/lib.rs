pub mod config;
pub mod diffusion;
pub mod qos;
pub mod server;

// Re-export shared infrastructure so downstream crates can
// `use image_gen_inference::*`.
pub use tangle_inference_core::{
    detect_gpus, parse_nvidia_smi_output, AppState, AppStateBuilder, BillingClient, CostModel,
    CostParams, GpuInfo, NonceStore, PerImageCostModel, RequestGuard, SpendAuthPayload,
};
pub use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, settle_billing, validate_spend_auth,
};
pub use tangle_inference_core::metrics;
pub use tangle_inference_core::billing;

use blueprint_sdk::std::sync::{Arc, OnceLock};
use blueprint_sdk::std::time::Duration;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::oneshot;

use crate::config::OperatorConfig;
use crate::diffusion::DiffusionBackend;
use crate::server::ImageGenBackend;

// --- ABI types for on-chain job encoding ---

sol! {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Input payload ABI-encoded in the Tangle job call.
    struct ImageGenRequest {
        string prompt;
        uint32 width;
        uint32 height;
        uint32 steps;
        uint32 numImages;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Output payload ABI-encoded in the Tangle job result.
    struct ImageGenResult {
        /// IPFS CID or URL where the generated images are stored
        string imageUri;
        uint32 numImages;
        uint32 widthUsed;
        uint32 heightUsed;
    }
}

// --- Job IDs ---

pub const IMAGE_GEN_JOB: u8 = 0;

// --- Shared state for the on-chain job handler ---

static DIFFUSION_ENDPOINT: OnceLock<DiffusionEndpoint> = OnceLock::new();

struct DiffusionEndpoint {
    url: String,
    model: String,
    default_steps: u32,
    client: reqwest::Client,
}

#[allow(clippy::result_large_err)]
fn register_diffusion_endpoint(config: &OperatorConfig) -> Result<(), RunnerError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.diffusion.generation_timeout_secs))
        .build()
        .map_err(|e| RunnerError::Other(format!("failed to build HTTP client: {e}").into()))?;
    let endpoint = DiffusionEndpoint {
        url: format!("{}/generate", config.diffusion.endpoint),
        model: config.diffusion.model.clone(),
        default_steps: config.diffusion.default_steps,
        client,
    };
    let _ = DIFFUSION_ENDPOINT.set(endpoint);
    Ok(())
}

/// Initialize the diffusion endpoint for testing without starting the server.
pub fn init_for_testing(base_url: &str, model: &str) {
    let endpoint = DiffusionEndpoint {
        url: format!("{base_url}/generate"),
        model: model.to_string(),
        default_steps: 30,
        client: reqwest::Client::new(),
    };
    let _ = DIFFUSION_ENDPOINT.set(endpoint);
}

// --- Router ---

pub fn router() -> Router {
    Router::new().route(
        IMAGE_GEN_JOB,
        run_image_gen
            .layer(TangleLayer)
            .layer(blueprint_sdk::tee::TeeLayer::new()),
    )
}

/// Direct image generation call -- same logic as run_image_gen but without TangleArg.
pub async fn generate_image_direct(
    request: &ImageGenRequest,
) -> Result<ImageGenResult, RunnerError> {
    let endpoint = DIFFUSION_ENDPOINT
        .get()
        .ok_or_else(|| RunnerError::Other("diffusion endpoint not registered".into()))?;

    let steps = if request.steps == 0 {
        endpoint.default_steps
    } else {
        request.steps
    };

    let body = serde_json::json!({
        "prompt": request.prompt,
        "model": endpoint.model,
        "width": request.width,
        "height": request.height,
        "steps": steps,
        "n": request.numImages.max(1),
        "cfg_scale": 7.5,
    });

    let resp = endpoint
        .client
        .post(&endpoint.url)
        .json(&body)
        .send()
        .await
        .map_err(|e| RunnerError::Other(format!("diffusion request failed: {e}").into()))?;

    let result: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| RunnerError::Other(format!("diffusion response parse failed: {e}").into()))?;

    let image_uri = result["image_uri"].as_str().unwrap_or("pending").to_string();
    let num_images = result["images"]
        .as_array()
        .map(|a| a.len() as u32)
        .unwrap_or(request.numImages.max(1));

    Ok(ImageGenResult {
        imageUri: image_uri,
        numImages: num_images,
        widthUsed: request.width,
        heightUsed: request.height,
    })
}

// --- Job handler ---

/// Handle an image generation job submitted on-chain.
#[debug_job]
pub async fn run_image_gen(
    TangleArg(request): TangleArg<ImageGenRequest>,
) -> Result<TangleResult<ImageGenResult>, RunnerError> {
    let result = generate_image_direct(&request).await?;
    Ok(TangleResult(result))
}

// --- Background service: HTTP server + diffusion health monitoring ---

/// Runs the diffusion health monitor and the OpenAI-compatible HTTP API as a
/// [`BackgroundService`]. Starts before the BlueprintRunner begins polling for
/// on-chain jobs.
#[derive(Clone)]
pub struct ImageGenServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for ImageGenServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            // 1. Create diffusion backend client and wait for readiness.
            let diffusion = match DiffusionBackend::new(config.clone()) {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    tracing::error!(error = %e, "failed to create diffusion backend");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            tracing::info!(
                endpoint = %config.diffusion.endpoint,
                "waiting for diffusion backend readiness"
            );
            if let Err(e) = diffusion.wait_ready(Duration::from_secs(120)).await {
                tracing::error!(error = %e, "diffusion backend failed to become ready");
                let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                return;
            }
            tracing::info!("diffusion backend is ready");

            if let Err(e) = register_diffusion_endpoint(&config) {
                tracing::error!(error = %e, "failed to register diffusion endpoint");
                let _ = tx.send(Err(e));
                return;
            }

            // 2. Build billing client.
            let billing_client = match BillingClient::new(&config.tangle, &config.billing) {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    tracing::error!(error = %e, "failed to create billing client");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            // 3. Create shutdown channel.
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            // 4. Build AppState via shared builder, attaching the image-gen backend.
            let operator_address = billing_client.operator_address();
            let nonce_store = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));
            let backend = ImageGenBackend::new(config.clone(), diffusion.clone());

            let state = match AppStateBuilder::new()
                .billing(billing_client)
                .nonce_store(nonce_store)
                .server_config(Arc::new(config.server.clone()))
                .billing_config(Arc::new(config.billing.clone()))
                .tangle_config(Arc::new(config.tangle.clone()))
                .operator_address(operator_address)
                .backend(backend)
                .build()
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "failed to build AppState");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            match server::start(state, shutdown_rx).await {
                Ok(_join_handle) => {
                    tracing::info!("HTTP server started -- background service ready");
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            }

            // 5. Health watchdog loop.
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(30)) => {}
                    _ = tokio::signal::ctrl_c() => {
                        tracing::info!("received shutdown signal");
                        let _ = shutdown_tx.send(true);
                        return;
                    }
                }

                if !diffusion.is_healthy().await {
                    tracing::warn!(
                        endpoint = %config.diffusion.endpoint,
                        "diffusion backend health check failed"
                    );
                }
            }
        });

        Ok(rx)
    }
}
