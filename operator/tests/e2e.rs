use std::sync::Arc;

use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use image_gen_inference::config::{
    BillingConfig, DiffusionConfig, GpuConfig, OperatorConfig, ServerConfig, TangleConfig,
};
use image_gen_inference::diffusion::DiffusionBackend;

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn test_config(diffusion_port: u16) -> OperatorConfig {
    OperatorConfig {
        tangle: TangleConfig {
            rpc_url: "http://localhost:8545".into(),
            chain_id: 31337,
            operator_key: "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
                .into(),
            tangle_core: "0x0000000000000000000000000000000000000000".into(),
            shielded_credits: "0x0000000000000000000000000000000000000000".into(),
            blueprint_id: 1,
            service_id: Some(1),
        },
        diffusion: DiffusionConfig {
            model: "test-model".into(),
            endpoint: format!("http://127.0.0.1:{diffusion_port}"),
            default_steps: 10,
            default_width: 1024,
            default_height: 1024,
            supported_resolutions: vec![
                "512x512".into(),
                "1024x1024".into(),
            ],
            generation_timeout_secs: 30,
            max_images: 4,
            supported_operations: vec![
                "generate".into(),
                "edit".into(),
                "variation".into(),
            ],
            max_image_size_bytes: 20 * 1024 * 1024,
        },
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0, // overridden per-test
            max_concurrent_requests: 8,
            max_request_body_bytes: 32 * 1024 * 1024,
            request_timeout_secs: 30,
            max_per_account_requests: 0,
        },
        billing: BillingConfig {
            required: false,
            price_per_image: 50000,
            price_per_extra_megapixel: 0,
            max_spend_per_request: 1_000_000,
            min_credit_balance: 1000,
            billing_required: false,
            min_charge_amount: 0,
            claim_max_retries: 3,
            clock_skew_tolerance_secs: 30,
            max_gas_price_gwei: 0,
            nonce_store_path: None,
            payment_token_address: None,
        },
        gpu: GpuConfig {
            expected_gpu_count: 0,
            min_vram_mib: 0,
            gpu_model: None,
            monitor_interval_secs: 30,
        },
        qos: None,
    }
}

async fn start_test_server(
    diffusion_port: u16,
) -> (u16, tokio::sync::watch::Sender<bool>, tokio::task::JoinHandle<()>) {
    let server_port = free_port();
    let mut config = test_config(diffusion_port);
    config.server.port = server_port;
    let config = Arc::new(config);

    let backend = Arc::new(DiffusionBackend::new(config.clone()).unwrap());

    let state = image_gen_inference::server::AppState {
        config,
        backend,
    };

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let handle = image_gen_inference::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

// -- Tests --

#[tokio::test]
async fn test_health_check_healthy_backend() {
    let mock_backend = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "ok"})))
        .mount(&mock_backend)
        .await;

    let (port, _tx, _h) = start_test_server(mock_backend.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["model"], "test-model");
}

#[tokio::test]
async fn test_health_check_unhealthy_backend() {
    let mock_backend = MockServer::start().await;
    // no health mock -> 404 from wiremock = unhealthy

    let (port, _tx, _h) = start_test_server(mock_backend.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn test_create_image_success() {
    let mock_backend = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "ok"})))
        .mount(&mock_backend)
        .await;

    Mock::given(method("POST"))
        .and(path("/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "images": [{"b64_json": "dGVzdA==", "revised_prompt": null}],
            "generation_time_ms": 1234
        })))
        .mount(&mock_backend)
        .await;

    let (port, _tx, _h) = start_test_server(mock_backend.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/images/generations"))
        .json(&serde_json::json!({
            "prompt": "a cat",
            "size": "1024x1024",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["created"].as_u64().is_some());
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert!(body["data"][0]["b64_json"].as_str().is_some());
}

#[tokio::test]
async fn test_create_image_invalid_size() {
    let mock_backend = MockServer::start().await;

    let (port, _tx, _h) = start_test_server(mock_backend.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/images/generations"))
        .json(&serde_json::json!({
            "prompt": "a cat",
            "size": "9999x9999",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("unsupported resolution"));
}

#[tokio::test]
async fn test_edit_image_success() {
    let mock_backend = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "ok"})))
        .mount(&mock_backend)
        .await;

    Mock::given(method("POST"))
        .and(path("/edit"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "images": [{"b64_json": "ZWRpdGVk", "revised_prompt": null}],
            "generation_time_ms": 2000
        })))
        .mount(&mock_backend)
        .await;

    let (port, _tx, _h) = start_test_server(mock_backend.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/images/edits"))
        .json(&serde_json::json!({
            "image": "dGVzdA==",
            "prompt": "make it blue",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert!(body["data"][0]["b64_json"].as_str().is_some());
}

#[tokio::test]
async fn test_list_models() {
    let mock_backend = MockServer::start().await;

    let (port, _tx, _h) = start_test_server(mock_backend.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/v1/models"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "test-model");
}
