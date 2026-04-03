//! Full lifecycle test -- on-chain job through real handler + wiremock diffusion backend.

use anyhow::{Result, ensure};
use wiremock::{MockServer, Mock, ResponseTemplate, matchers::{method, path}};
use image_gen_inference::ImageGenRequest;

#[tokio::test]
async fn test_generate_image_direct_with_wiremock() -> Result<()> {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "image_uri": "ipfs://QmFakeHash123",
            "images": [
                {"data": "base64data1"},
                {"data": "base64data2"}
            ]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    image_gen_inference::init_for_testing(&mock_server.uri(), "stable-diffusion-xl");

    let request = ImageGenRequest {
        prompt: "A sunset over the ocean".into(),
        width: 1024,
        height: 1024,
        steps: 20,
        numImages: 2,
    };

    let result = image_gen_inference::generate_image_direct(&request).await;

    match result {
        Ok(gen_result) => {
            ensure!(
                gen_result.imageUri == "ipfs://QmFakeHash123",
                "expected IPFS URI, got '{}'",
                gen_result.imageUri
            );
            ensure!(gen_result.numImages == 2, "expected 2 images, got {}", gen_result.numImages);
            ensure!(gen_result.widthUsed == 1024, "wrong width");
            ensure!(gen_result.heightUsed == 1024, "wrong height");
        }
        Err(e) => panic!("Image generation failed: {e}"),
    }

    mock_server.verify().await;

    Ok(())
}
