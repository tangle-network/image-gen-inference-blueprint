use image_gen_inference::diffusion::{
    EditParams, GenerateParams, GenerateResult, GeneratedImage, VariationParams,
};

#[test]
fn generate_params_serialization_roundtrip() {
    let params = GenerateParams {
        prompt: "a cat in space".to_string(),
        negative_prompt: Some("blurry, low quality".to_string()),
        model: Some("stable-diffusion-xl".to_string()),
        width: 1024,
        height: 1024,
        steps: 30,
        n: 2,
        seed: Some(42),
        cfg_scale: 7.5,
    };

    let json = serde_json::to_string(&params).unwrap();
    let deserialized: GenerateParams = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.prompt, "a cat in space");
    assert_eq!(deserialized.negative_prompt.as_deref(), Some("blurry, low quality"));
    assert_eq!(deserialized.width, 1024);
    assert_eq!(deserialized.height, 1024);
    assert_eq!(deserialized.steps, 30);
    assert_eq!(deserialized.n, 2);
    assert_eq!(deserialized.seed, Some(42));
    assert!((deserialized.cfg_scale - 7.5).abs() < f32::EPSILON);
}

#[test]
fn generate_params_defaults() {
    let json = r#"{"prompt":"test","width":512,"height":512,"steps":20}"#;
    let params: GenerateParams = serde_json::from_str(json).unwrap();

    assert_eq!(params.n, 1); // default_n
    assert!((params.cfg_scale - 7.5).abs() < f32::EPSILON); // default_cfg_scale
    assert!(params.negative_prompt.is_none());
    assert!(params.model.is_none());
    assert!(params.seed.is_none());
}

#[test]
fn edit_params_serialization() {
    let params = EditParams {
        prompt: "make it red".to_string(),
        negative_prompt: None,
        model: None,
        n: 1,
        size: Some("1024x1024".to_string()),
        seed: None,
        cfg_scale: 7.5,
    };

    let json = serde_json::to_string(&params).unwrap();
    let deserialized: EditParams = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.prompt, "make it red");
    assert_eq!(deserialized.size.as_deref(), Some("1024x1024"));
}

#[test]
fn variation_params_serialization() {
    let params = VariationParams {
        model: Some("sdxl".to_string()),
        n: 3,
        size: None,
        seed: Some(123),
    };

    let json = serde_json::to_string(&params).unwrap();
    let deserialized: VariationParams = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.model.as_deref(), Some("sdxl"));
    assert_eq!(deserialized.n, 3);
    assert_eq!(deserialized.seed, Some(123));
}

#[test]
fn generate_result_deserialization() {
    let json = r#"{
        "images": [
            {"b64_json": "aW1hZ2VfZGF0YQ==", "revised_prompt": "a cat floating in outer space"},
            {"b64_json": "c2Vjb25kX2ltYWdl", "revised_prompt": null}
        ],
        "generation_time_ms": 5432
    }"#;

    let result: GenerateResult = serde_json::from_str(json).unwrap();

    assert_eq!(result.images.len(), 2);
    assert_eq!(result.images[0].b64_json, "aW1hZ2VfZGF0YQ==");
    assert_eq!(
        result.images[0].revised_prompt.as_deref(),
        Some("a cat floating in outer space")
    );
    assert!(result.images[1].revised_prompt.is_none());
    assert_eq!(result.generation_time_ms, Some(5432));
}

#[test]
fn generated_image_minimal() {
    let img = GeneratedImage {
        b64_json: "dGVzdA==".to_string(),
        revised_prompt: None,
    };

    let json = serde_json::to_string(&img).unwrap();
    assert!(json.contains("dGVzdA=="));

    let deserialized: GeneratedImage = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.b64_json, "dGVzdA==");
}
