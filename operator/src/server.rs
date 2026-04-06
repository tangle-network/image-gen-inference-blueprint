//! OpenAI-compatible Images HTTP server for the image-gen operator.
//!
//! All shared infrastructure (nonce store, spend-auth validation, x402 headers,
//! metrics, app state container) lives in `tangle-inference-core`. This module
//! only contains:
//!
//! * `ImageGenBackend` — the backend attached to `AppState` via `AppStateBuilder`.
//! * Request/response types for the OpenAI Images wire format.
//! * HTTP handlers that glue the shared billing flow to the diffusion backend.

use std::collections::HashMap;

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use tangle_inference_core::server::{
    acquire_permit, billing_gate, error_response, gpu_health_handler, metrics_handler,
    settle_billing,
};
use tangle_inference_core::{
    detect_gpus, AppState, CostModel, CostParams, PerImageCostModel, RequestGuard,
    SpendAuthPayload,
};

use crate::config::OperatorConfig;
use crate::diffusion::{DiffusionBackend, EditParams, GenerateParams, GeneratedImage, VariationParams};

/// Backend attached to `AppState` via `AppStateBuilder::backend`. Handlers
/// retrieve it via `state.backend::<ImageGenBackend>().unwrap()`.
pub struct ImageGenBackend {
    pub config: Arc<OperatorConfig>,
    pub diffusion: Arc<DiffusionBackend>,
    pub cost_model: Arc<PerImageCostModel>,
}

impl ImageGenBackend {
    pub fn new(config: Arc<OperatorConfig>, diffusion: Arc<DiffusionBackend>) -> Self {
        let cost_model = Arc::new(PerImageCostModel {
            price_per_image: config.diffusion.price_per_image,
        });
        Self {
            config,
            diffusion,
            cost_model,
        }
    }

    /// Calculate the cost for `n` images.
    pub fn calculate_cost(&self, n_images: u32) -> u64 {
        let mut extra = HashMap::new();
        extra.insert("images".to_string(), n_images as u64);
        self.cost_model.calculate_cost(&CostParams {
            extra,
            ..Default::default()
        })
    }
}

fn backend_from(state: &AppState) -> &ImageGenBackend {
    state
        .backend::<ImageGenBackend>()
        .expect("AppState backend is ImageGenBackend (checked in server::start)")
}

/// Start the HTTP server with graceful shutdown support, returns a join handle.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let backend = state
        .backend::<ImageGenBackend>()
        .ok_or_else(|| anyhow::anyhow!("AppState backend is not an ImageGenBackend"))?;

    // Account for base64-encoded uploads (4/3 overhead) plus JSON wrapper.
    let image_limit = backend.config.diffusion.max_image_size_bytes * 4 / 3 + 4096;
    let max_body = state.server_config.max_request_body_bytes.max(image_limit);
    let request_timeout = Duration::from_secs(state.server_config.stream_timeout_secs);
    let bind = format!("{}:{}", state.server_config.host, state.server_config.port);

    let app = HttpRouter::new()
        .route("/v1/images/generations", post(create_image))
        .route("/v1/images/edits", post(edit_image))
        .route("/v1/images/variations", post(create_variation))
        .route("/v1/models", get(list_models))
        .route("/v1/operator", get(operator_info))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health_handler))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(max_body))
        .layer(TimeoutLayer::new(request_timeout))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        let shutdown_signal = async move {
            let _ = shutdown_rx.wait_for(|&v| v).await;
            tracing::info!("HTTP server received shutdown signal, draining connections");
        };
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal)
            .await
        {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// --- OpenAI-compatible request/response types ---

#[derive(Debug, Deserialize)]
pub struct CreateImageRequest {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_n")]
    pub n: u32,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    #[serde(default = "default_quality")]
    pub quality: String,
    #[allow(dead_code)]
    #[serde(default = "default_style")]
    pub style: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    /// ShieldedCredits spend authorization (also accepted via x402 headers).
    #[serde(default)]
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Deserialize)]
pub struct EditImageRequest {
    pub image: String,
    pub prompt: String,
    #[serde(default)]
    pub mask: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_n")]
    pub n: u32,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Deserialize)]
pub struct CreateVariationRequest {
    pub image: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_n")]
    pub n: u32,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Serialize)]
pub struct CreateImageResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Serialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

fn default_n() -> u32 {
    1
}
fn default_size() -> String {
    "1024x1024".to_string()
}
fn default_response_format() -> String {
    "b64_json".to_string()
}
fn default_quality() -> String {
    "standard".to_string()
}
fn default_style() -> String {
    "vivid".to_string()
}

fn parse_size(size: &str) -> Option<(u32, u32)> {
    let (w, h) = size.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

fn build_image_response(images: Vec<GeneratedImage>, response_format: &str) -> Response {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let data: Vec<ImageData> = images
        .into_iter()
        .map(|img| {
            if response_format == "b64_json" {
                ImageData {
                    b64_json: Some(img.b64_json),
                    url: None,
                    revised_prompt: img.revised_prompt,
                }
            } else {
                ImageData {
                    b64_json: None,
                    url: Some("not_implemented".to_string()),
                    revised_prompt: img.revised_prompt,
                }
            }
        })
        .collect();

    Json(CreateImageResponse { created, data }).into_response()
}

fn require_operation(backend: &ImageGenBackend, operation: &str) -> Option<Response> {
    if !backend
        .config
        .diffusion
        .supported_operations
        .iter()
        .any(|op| op == operation)
    {
        Some(error_response(
            StatusCode::NOT_IMPLEMENTED,
            format!("this operator does not support '{operation}' -- backend lacks capability"),
            "not_implemented",
            "operation_not_supported",
        ))
    } else {
        None
    }
}

fn validate_image_size(backend: &ImageGenBackend, b64: &str) -> Option<Response> {
    let estimated_bytes = b64.len() * 3 / 4;
    let max = backend.config.diffusion.max_image_size_bytes;
    if estimated_bytes > max {
        Some(error_response(
            StatusCode::BAD_REQUEST,
            format!("uploaded image exceeds maximum size ({estimated_bytes} > {max} bytes)"),
            "invalid_request_error",
            "image_too_large",
        ))
    } else {
        None
    }
}

// --- Handlers ---

async fn create_image(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<CreateImageRequest>,
) -> Response {
    let backend = backend_from(&state);
    let model_name = req.model.clone().unwrap_or_else(|| backend.config.diffusion.model.clone());
    let mut metrics_guard = RequestGuard::new(&model_name);

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // Parse/validate size.
    let (width, height) = match parse_size(&req.size) {
        Some(dims) => dims,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "invalid size format '{}', expected 'WxH' (e.g. '1024x1024')",
                    req.size
                ),
                "invalid_request_error",
                "invalid_size",
            );
        }
    };

    let size_str = format!("{width}x{height}");
    if !backend
        .config
        .diffusion
        .supported_resolutions
        .contains(&size_str)
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported resolution '{size_str}'. Supported: {:?}",
                backend.config.diffusion.supported_resolutions
            ),
            "invalid_request_error",
            "unsupported_resolution",
        );
    }

    // Clamp n and resolve steps.
    let n = req.n.clamp(1, backend.config.diffusion.max_images);
    let steps = req.steps.unwrap_or(if req.quality == "hd" {
        backend.config.diffusion.default_steps * 2
    } else {
        backend.config.diffusion.default_steps
    });

    // Billing gate: extract x402 -> validate -> authorize on-chain.
    let estimated_cost = backend.calculate_cost(n);
    let (spend_auth, preauth) =
        match billing_gate(&state, &headers, req.spend_auth.take(), estimated_cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    // Check backend health before serving (only when billing).
    if spend_auth.is_some() && !backend.diffusion.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "diffusion backend is unavailable — billing not initiated".to_string(),
            "upstream_error",
            "backend_unhealthy",
        );
    }

    // Generate.
    let params = GenerateParams {
        prompt: req.prompt,
        negative_prompt: req.negative_prompt,
        model: req.model,
        width,
        height,
        steps,
        n,
        seed: req.seed,
        cfg_scale: req.cfg_scale.unwrap_or(7.5),
    };

    let result = match backend.diffusion.generate(&params).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "image generation failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("upstream diffusion error: {e}"),
                "upstream_error",
                "generation_failed",
            );
        }
    };

    let n_actual = result.images.len() as u32;
    metrics_guard.set_success();

    // Settle billing.
    if let Some(ref auth) = spend_auth {
        let actual_cost = backend.calculate_cost(n_actual);
        if let Err(e) = settle_billing(&state.billing, auth, preauth.unwrap_or(0), actual_cost).await {
            tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
        }
    }

    build_image_response(result.images, &req.response_format)
}

async fn edit_image(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<EditImageRequest>,
) -> Response {
    let backend = backend_from(&state);
    let model_name = req.model.clone().unwrap_or_else(|| backend.config.diffusion.model.clone());
    let mut metrics_guard = RequestGuard::new(&model_name);

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    if let Some(err) = require_operation(backend, "edit") {
        return err;
    }

    if let Some(err) = validate_image_size(backend, &req.image) {
        return err;
    }
    if let Some(ref mask) = req.mask {
        if let Some(err) = validate_image_size(backend, mask) {
            return err;
        }
    }

    let n = req.n.clamp(1, backend.config.diffusion.max_images);

    let estimated_cost = backend.calculate_cost(n);
    let (spend_auth, preauth) =
        match billing_gate(&state, &headers, req.spend_auth.take(), estimated_cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    if spend_auth.is_some() && !backend.diffusion.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "diffusion backend is unavailable — billing not initiated".to_string(),
            "upstream_error",
            "backend_unhealthy",
        );
    }

    let params = EditParams {
        prompt: req.prompt.clone(),
        negative_prompt: req.negative_prompt,
        model: req.model,
        n,
        size: req.size,
        seed: req.seed,
        cfg_scale: req.cfg_scale.unwrap_or(7.5),
    };

    let result = match backend
        .diffusion
        .edit(&req.image, req.mask.as_deref(), &req.prompt, &params)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "image edit failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("upstream diffusion error: {e}"),
                "upstream_error",
                "edit_failed",
            );
        }
    };

    let n_actual = result.images.len() as u32;
    metrics_guard.set_success();

    if let Some(ref auth) = spend_auth {
        let actual_cost = backend.calculate_cost(n_actual);
        if let Err(e) = settle_billing(&state.billing, auth, preauth.unwrap_or(0), actual_cost).await {
            tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
        }
    }

    build_image_response(result.images, &req.response_format)
}

async fn create_variation(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<CreateVariationRequest>,
) -> Response {
    let backend = backend_from(&state);
    let model_name = req.model.clone().unwrap_or_else(|| backend.config.diffusion.model.clone());
    let mut metrics_guard = RequestGuard::new(&model_name);

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    if let Some(err) = require_operation(backend, "variation") {
        return err;
    }

    if let Some(err) = validate_image_size(backend, &req.image) {
        return err;
    }

    let n = req.n.clamp(1, backend.config.diffusion.max_images);

    let estimated_cost = backend.calculate_cost(n);
    let (spend_auth, preauth) =
        match billing_gate(&state, &headers, req.spend_auth.take(), estimated_cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    if spend_auth.is_some() && !backend.diffusion.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "diffusion backend is unavailable — billing not initiated".to_string(),
            "upstream_error",
            "backend_unhealthy",
        );
    }

    let params = VariationParams {
        model: req.model,
        n,
        size: req.size,
        seed: req.seed,
    };

    let result = match backend.diffusion.variation(&req.image, &params).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "image variation failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("upstream diffusion error: {e}"),
                "upstream_error",
                "variation_failed",
            );
        }
    };

    let n_actual = result.images.len() as u32;
    metrics_guard.set_success();

    if let Some(ref auth) = spend_auth {
        let actual_cost = backend.calculate_cost(n_actual);
        if let Err(e) = settle_billing(&state.billing, auth, preauth.unwrap_or(0), actual_cost).await {
            tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
        }
    }

    build_image_response(result.images, &req.response_format)
}

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let backend = backend_from(&state);
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: backend.config.diffusion.model.clone(),
            object: "model".to_string(),
            owned_by: "operator".to_string(),
        }],
    })
}

async fn operator_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    let gpu_info = detect_gpus().await.unwrap_or_default();
    Json(serde_json::json!({
        "operator": format!("{:#x}", state.operator_address),
        "model": backend.config.diffusion.model,
        "type": "image-generation",
        "pricing": {
            "price_per_image": backend.config.diffusion.price_per_image,
            "currency": "tsUSD",
        },
        "capabilities": {
            "supported_resolutions": backend.config.diffusion.supported_resolutions,
            "max_images_per_request": backend.config.diffusion.max_images,
            "default_steps": backend.config.diffusion.default_steps,
            "supported_operations": backend.config.diffusion.supported_operations,
            "max_image_size_bytes": backend.config.diffusion.max_image_size_bytes,
        },
        "gpu": {
            "count": backend.config.gpu.expected_gpu_count,
            "min_vram_mib": backend.config.gpu.min_vram_mib,
            "model": backend.config.gpu.gpu_model,
            "detected": gpu_info,
        },
        "server": {
            "max_concurrent_requests": state.server_config.max_concurrent_requests,
        },
        "billing_required": state.billing_config.billing_required,
        "payment_token": state.billing_config.payment_token_address,
    }))
}

async fn health_check(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let backend = backend_from(&state);
    if backend.diffusion.is_healthy().await {
        Ok(Json(serde_json::json!({
            "status": "ok",
            "model": backend.config.diffusion.model,
            "backend": backend.diffusion.endpoint(),
        })))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}
