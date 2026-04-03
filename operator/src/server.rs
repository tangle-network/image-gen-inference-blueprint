use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::config::OperatorConfig;
use crate::diffusion::{DiffusionBackend, EditParams, GenerateParams, GeneratedImage, VariationParams};
use crate::health;

// --- Application state ---

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<OperatorConfig>,
    pub backend: Arc<DiffusionBackend>,
}

/// Start the HTTP server with graceful shutdown support.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let app = HttpRouter::new()
        .route("/v1/images/generations", post(create_image))
        .route("/v1/images/edits", post(edit_image))
        .route("/v1/images/variations", post(create_variation))
        .route("/v1/models", get(list_models))
        .route("/v1/operator", get(operator_info))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health))
        .layer(DefaultBodyLimit::max({
            // Account for base64-encoded images (4/3 overhead) plus JSON wrapper
            let image_limit = state.config.diffusion.max_image_size_bytes * 4 / 3 + 4096;
            state.config.server.max_request_body_bytes.max(image_limit)
        }))
        .layer(TimeoutLayer::new(Duration::from_secs(
            state.config.server.request_timeout_secs,
        )))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let bind = format!("{}:{}", state.config.server.host, state.config.server.port);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        let shutdown_signal = async move {
            let _ = shutdown_rx.wait_for(|&v| v).await;
            tracing::info!("HTTP server received shutdown signal");
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

/// OpenAI Images API compatible request.
/// See: https://platform.openai.com/docs/api-reference/images/create
#[derive(Debug, Deserialize)]
pub struct CreateImageRequest {
    /// Text prompt describing the image to generate
    pub prompt: String,

    /// Model to use (optional, defaults to config)
    #[serde(default)]
    pub model: Option<String>,

    /// Number of images to generate (1-max_images)
    #[serde(default = "default_n")]
    pub n: u32,

    /// Image size as "WxH" string (e.g. "1024x1024")
    #[serde(default = "default_size")]
    pub size: String,

    /// Response format: "b64_json" or "url"
    #[serde(default = "default_response_format")]
    pub response_format: String,

    /// Quality hint: "standard" or "hd"
    #[serde(default = "default_quality")]
    pub quality: String,

    /// Style hint: "vivid" or "natural"
    #[serde(default = "default_style")]
    pub style: String,

    // Extension fields (not in OpenAI spec but useful for diffusion)

    /// Negative prompt
    #[serde(default)]
    pub negative_prompt: Option<String>,

    /// Number of inference steps (0 = use default)
    #[serde(default)]
    pub steps: Option<u32>,

    /// Random seed
    #[serde(default)]
    pub seed: Option<i64>,

    /// CFG / guidance scale
    #[serde(default)]
    pub cfg_scale: Option<f32>,

    /// SpendAuth for billing
    #[serde(default)]
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Deserialize)]
pub struct SpendAuthPayload {
    pub commitment: String,
    pub service_id: u64,
    pub job_index: u8,
    pub amount: String,
    pub operator: String,
    pub nonce: u64,
    pub expiry: u64,
    pub signature: String,
}

/// OpenAI Images API compatible edit request.
/// See: https://platform.openai.com/docs/api-reference/images/createEdit
#[derive(Debug, Deserialize)]
pub struct EditImageRequest {
    /// Base64-encoded source image
    pub image: String,

    /// Text prompt describing the desired edit
    pub prompt: String,

    /// Base64-encoded mask image (white = edit region). Optional for img2img.
    #[serde(default)]
    pub mask: Option<String>,

    /// Model to use (optional, defaults to config)
    #[serde(default)]
    pub model: Option<String>,

    /// Number of images to generate
    #[serde(default = "default_n")]
    pub n: u32,

    /// Image size as "WxH" string
    #[serde(default)]
    pub size: Option<String>,

    /// Response format: "b64_json" or "url"
    #[serde(default = "default_response_format")]
    pub response_format: String,

    // Extension fields

    #[serde(default)]
    pub negative_prompt: Option<String>,

    #[serde(default)]
    pub seed: Option<i64>,

    #[serde(default)]
    pub cfg_scale: Option<f32>,

    #[serde(default)]
    pub spend_auth: Option<SpendAuthPayload>,
}

/// OpenAI Images API compatible variation request.
/// See: https://platform.openai.com/docs/api-reference/images/createVariation
#[derive(Debug, Deserialize)]
pub struct CreateVariationRequest {
    /// Base64-encoded source image
    pub image: String,

    /// Model to use (optional, defaults to config)
    #[serde(default)]
    pub model: Option<String>,

    /// Number of variations to generate
    #[serde(default = "default_n")]
    pub n: u32,

    /// Image size as "WxH" string
    #[serde(default)]
    pub size: Option<String>,

    /// Response format: "b64_json" or "url"
    #[serde(default = "default_response_format")]
    pub response_format: String,

    // Extension fields

    #[serde(default)]
    pub seed: Option<i64>,

    #[serde(default)]
    pub spend_auth: Option<SpendAuthPayload>,
}

/// OpenAI Images API compatible response.
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

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    code: String,
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

fn error_response(status: StatusCode, message: String, error_type: &str, code: &str) -> Response {
    let body = ErrorResponse {
        error: ErrorDetail {
            message,
            r#type: error_type.to_string(),
            code: code.to_string(),
        },
    };
    (status, Json(body)).into_response()
}

/// Parse "WxH" size string into (width, height).
fn parse_size(size: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() != 2 {
        return None;
    }
    let w = parts[0].parse().ok()?;
    let h = parts[1].parse().ok()?;
    Some((w, h))
}

// --- Handlers ---

async fn create_image(
    State(state): State<AppState>,
    _headers: HeaderMap,
    Json(req): Json<CreateImageRequest>,
) -> Response {
    // 1. Parse and validate size
    let (width, height) = match parse_size(&req.size) {
        Some(dims) => dims,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid size format '{}', expected 'WxH' (e.g. '1024x1024')", req.size),
                "invalid_request_error",
                "invalid_size",
            );
        }
    };

    // Validate resolution is supported
    let size_str = format!("{width}x{height}");
    if !state.config.diffusion.supported_resolutions.contains(&size_str) {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported resolution '{size_str}'. Supported: {:?}",
                state.config.diffusion.supported_resolutions
            ),
            "invalid_request_error",
            "unsupported_resolution",
        );
    }

    // 2. Validate n
    let n = req.n.max(1).min(state.config.diffusion.max_images);

    // 3. Determine steps
    let steps = req.steps.unwrap_or(if req.quality == "hd" {
        state.config.diffusion.default_steps * 2
    } else {
        state.config.diffusion.default_steps
    });

    // 4. Check backend health before processing
    if !state.backend.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "diffusion backend is unavailable".to_string(),
            "upstream_error",
            "backend_unhealthy",
        );
    }

    // 5. Generate
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

    let result = match state.backend.generate(&params).await {
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

    // 6. Build response
    build_image_response(result.images, &req.response_format)
}

/// Check that the backend supports a given operation, returning a 501 error response if not.
fn require_operation(state: &AppState, operation: &str) -> Option<Response> {
    if !state.config.diffusion.supported_operations.contains(&operation.to_string()) {
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

/// Validate uploaded image size against config limit.
fn validate_image_size(state: &AppState, b64: &str) -> Option<Response> {
    // Base64 encodes 3 bytes per 4 chars; estimate decoded size
    let estimated_bytes = b64.len() * 3 / 4;
    let max = state.config.diffusion.max_image_size_bytes;
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

/// Build a CreateImageResponse from GenerateResult images.
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

async fn edit_image(
    State(state): State<AppState>,
    _headers: HeaderMap,
    Json(req): Json<EditImageRequest>,
) -> Response {
    // Check operation support
    if let Some(err) = require_operation(&state, "edit") {
        return err;
    }

    // Validate image size
    if let Some(err) = validate_image_size(&state, &req.image) {
        return err;
    }
    if let Some(ref mask) = req.mask {
        if let Some(err) = validate_image_size(&state, mask) {
            return err;
        }
    }

    // Validate n
    let n = req.n.max(1).min(state.config.diffusion.max_images);

    // Check backend health
    if !state.backend.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "diffusion backend is unavailable".to_string(),
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

    let result = match state
        .backend
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

    build_image_response(result.images, &req.response_format)
}

async fn create_variation(
    State(state): State<AppState>,
    _headers: HeaderMap,
    Json(req): Json<CreateVariationRequest>,
) -> Response {
    // Check operation support
    if let Some(err) = require_operation(&state, "variation") {
        return err;
    }

    // Validate image size
    if let Some(err) = validate_image_size(&state, &req.image) {
        return err;
    }

    // Validate n
    let n = req.n.max(1).min(state.config.diffusion.max_images);

    // Check backend health
    if !state.backend.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "diffusion backend is unavailable".to_string(),
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

    let result = match state.backend.variation(&req.image, &params).await {
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

    build_image_response(result.images, &req.response_format)
}

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.config.diffusion.model.clone(),
            object: "model".to_string(),
            owned_by: "operator".to_string(),
        }],
    })
}

async fn operator_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let gpu_info = health::detect_gpus().await.unwrap_or_default();
    Json(serde_json::json!({
        "model": state.config.diffusion.model,
        "type": "image-generation",
        "pricing": {
            "price_per_image": state.config.billing.price_per_image,
            "price_per_extra_megapixel": state.config.billing.price_per_extra_megapixel,
            "currency": "tsUSD",
        },
        "capabilities": {
            "supported_resolutions": state.config.diffusion.supported_resolutions,
            "max_images_per_request": state.config.diffusion.max_images,
            "default_steps": state.config.diffusion.default_steps,
            "supported_operations": state.config.diffusion.supported_operations,
            "max_image_size_bytes": state.config.diffusion.max_image_size_bytes,
        },
        "gpu": {
            "count": state.config.gpu.expected_gpu_count,
            "min_vram_mib": state.config.gpu.min_vram_mib,
            "model": state.config.gpu.gpu_model,
            "detected": gpu_info,
        },
        "server": {
            "max_concurrent_requests": state.config.server.max_concurrent_requests,
        },
        "billing_required": state.config.billing.billing_required,
        "payment_token": state.config.billing.payment_token_address,
    }))
}

async fn health_check(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let backend_healthy = state.backend.is_healthy().await;

    if backend_healthy {
        Ok(Json(serde_json::json!({
            "status": "ok",
            "model": state.config.diffusion.model,
            "backend": state.backend.endpoint(),
        })))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn gpu_health() -> Result<Json<Vec<health::GpuInfo>>, (StatusCode, String)> {
    match health::detect_gpus().await {
        Ok(gpus) => Ok(Json(gpus)),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}
