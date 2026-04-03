//! Diffusion backend client.
//!
//! Wraps an external diffusion server (ComfyUI, A1111, or diffusers HTTP
//! endpoint) similar to how the vLLM blueprint wraps a vLLM subprocess.
//! The operator does NOT run diffusion inference itself -- it proxies to
//! a configurable HTTP endpoint.

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::config::OperatorConfig;

/// Parameters for a single image generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateParams {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    pub model: Option<String>,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    /// Number of images to generate (default 1)
    #[serde(default = "default_n")]
    pub n: u32,
    /// Fixed-point seed. None = random.
    pub seed: Option<i64>,
    /// CFG scale / guidance scale
    #[serde(default = "default_cfg_scale")]
    pub cfg_scale: f32,
}

/// Parameters for an image edit (img2img / inpainting) request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditParams {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    pub model: Option<String>,
    /// Number of images to generate (default 1)
    #[serde(default = "default_n")]
    pub n: u32,
    /// Image size as "WxH" (optional, defaults to source image dimensions)
    pub size: Option<String>,
    /// Random seed
    pub seed: Option<i64>,
    /// CFG / guidance scale
    #[serde(default = "default_cfg_scale")]
    pub cfg_scale: f32,
}

/// Parameters for an image variation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationParams {
    pub model: Option<String>,
    /// Number of variations to generate (default 1)
    #[serde(default = "default_n")]
    pub n: u32,
    /// Output size as "WxH" (optional)
    pub size: Option<String>,
    /// Random seed
    pub seed: Option<i64>,
}

fn default_n() -> u32 {
    1
}

fn default_cfg_scale() -> f32 {
    7.5
}

/// Result of a single generated image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    /// Base64-encoded image bytes (PNG)
    pub b64_json: String,
    /// Revised prompt (if the backend modified it)
    pub revised_prompt: Option<String>,
}

/// Result of a generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResult {
    pub images: Vec<GeneratedImage>,
    /// Backend-reported generation time in milliseconds
    pub generation_time_ms: Option<u64>,
}

/// Manages the connection to an external diffusion server.
pub struct DiffusionBackend {
    config: Arc<OperatorConfig>,
    client: reqwest::Client,
    /// Track whether the backend was reachable on last check
    healthy: Mutex<bool>,
}

impl DiffusionBackend {
    /// Create a new backend client pointing at the configured endpoint.
    pub fn new(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        let timeout = Duration::from_secs(config.diffusion.generation_timeout_secs);
        let client = reqwest::Client::builder().timeout(timeout).build()?;
        Ok(Self {
            config,
            client,
            healthy: Mutex::new(false),
        })
    }

    /// Wait until the diffusion backend's health endpoint responds.
    pub async fn wait_ready(&self, timeout: Duration) -> anyhow::Result<()> {
        let health_url = format!("{}/health", self.config.diffusion.endpoint);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                anyhow::bail!(
                    "diffusion backend at {} failed to become ready within {}s",
                    self.config.diffusion.endpoint,
                    timeout.as_secs()
                );
            }

            match self
                .client
                .get(&health_url)
                .timeout(Duration::from_secs(5))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    *self.healthy.lock().await = true;
                    return Ok(());
                }
                _ => {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }

    /// Check if the diffusion backend is currently healthy.
    pub async fn is_healthy(&self) -> bool {
        let health_url = format!("{}/health", self.config.diffusion.endpoint);
        let result = matches!(
            self.client
                .get(&health_url)
                .timeout(Duration::from_secs(5))
                .send()
                .await,
            Ok(r) if r.status().is_success()
        );
        *self.healthy.lock().await = result;
        result
    }

    /// Generate images by proxying to the diffusion backend.
    ///
    /// The backend is expected to accept a JSON POST at /generate with:
    /// ```json
    /// {
    ///   "prompt": "...",
    ///   "negative_prompt": "...",
    ///   "model": "...",
    ///   "width": 1024,
    ///   "height": 1024,
    ///   "steps": 30,
    ///   "n": 1,
    ///   "seed": 42,
    ///   "cfg_scale": 7.5
    /// }
    /// ```
    ///
    /// And return:
    /// ```json
    /// {
    ///   "images": [{"b64_json": "...", "revised_prompt": null}],
    ///   "generation_time_ms": 5000
    /// }
    /// ```
    pub async fn generate(&self, params: &GenerateParams) -> anyhow::Result<GenerateResult> {
        let url = format!("{}/generate", self.config.diffusion.endpoint);

        let model = params
            .model
            .as_deref()
            .unwrap_or(&self.config.diffusion.model);

        let body = serde_json::json!({
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "model": model,
            "width": params.width,
            "height": params.height,
            "steps": params.steps,
            "n": params.n,
            "seed": params.seed,
            "cfg_scale": params.cfg_scale,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "diffusion backend request failed");
                anyhow::anyhow!("diffusion backend request failed: {e}")
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("diffusion backend returned {status}: {body}");
        }

        let result: GenerateResult = resp.json().await.map_err(|e| {
            tracing::error!(error = %e, "diffusion backend response parse failed");
            anyhow::anyhow!("diffusion backend response parse failed: {e}")
        })?;

        Ok(result)
    }

    /// Edit an image (img2img / inpainting) by proxying to the diffusion backend.
    ///
    /// Posts base64-encoded image and optional mask to DIFFUSION_ENDPOINT/edit.
    /// Returns the same GenerateResult format as generate().
    pub async fn edit(
        &self,
        image_b64: &str,
        mask_b64: Option<&str>,
        prompt: &str,
        params: &EditParams,
    ) -> anyhow::Result<GenerateResult> {
        let url = format!("{}/edit", self.config.diffusion.endpoint);

        let model = params
            .model
            .as_deref()
            .unwrap_or(&self.config.diffusion.model);

        let body = serde_json::json!({
            "image": image_b64,
            "mask": mask_b64,
            "prompt": prompt,
            "negative_prompt": params.negative_prompt,
            "model": model,
            "n": params.n,
            "size": params.size,
            "seed": params.seed,
            "cfg_scale": params.cfg_scale,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "diffusion edit request failed");
                anyhow::anyhow!("diffusion edit request failed: {e}")
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("diffusion backend edit returned {status}: {body}");
        }

        let result: GenerateResult = resp.json().await.map_err(|e| {
            tracing::error!(error = %e, "diffusion edit response parse failed");
            anyhow::anyhow!("diffusion edit response parse failed: {e}")
        })?;

        Ok(result)
    }

    /// Generate variations of an input image by proxying to the diffusion backend.
    ///
    /// Posts base64-encoded image to DIFFUSION_ENDPOINT/variation.
    /// Returns the same GenerateResult format as generate().
    pub async fn variation(
        &self,
        image_b64: &str,
        params: &VariationParams,
    ) -> anyhow::Result<GenerateResult> {
        let url = format!("{}/variation", self.config.diffusion.endpoint);

        let model = params
            .model
            .as_deref()
            .unwrap_or(&self.config.diffusion.model);

        let body = serde_json::json!({
            "image": image_b64,
            "model": model,
            "n": params.n,
            "size": params.size,
            "seed": params.seed,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "diffusion variation request failed");
                anyhow::anyhow!("diffusion variation request failed: {e}")
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("diffusion backend variation returned {status}: {body}");
        }

        let result: GenerateResult = resp.json().await.map_err(|e| {
            tracing::error!(error = %e, "diffusion variation response parse failed");
            anyhow::anyhow!("diffusion variation response parse failed: {e}")
        })?;

        Ok(result)
    }

    /// Get the endpoint URL for this backend.
    pub fn endpoint(&self) -> &str {
        &self.config.diffusion.endpoint
    }
}
