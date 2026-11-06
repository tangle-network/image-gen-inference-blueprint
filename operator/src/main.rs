use alloy_sol_types::SolValue;
use image_gen_inference::config::OperatorConfig;
use image_gen_inference::ImageGenServer;

fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let gpu_count = config.gpu.expected_gpu_count;
    let total_vram = config.gpu.min_vram_mib;
    let gpu_model = config
        .gpu
        .gpu_model
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let endpoint = format!("http://{}:{}", config.server.host, config.server.port);
    let resolutions = config.diffusion.supported_resolutions.clone();

    (
        config.diffusion.model.clone(),
        gpu_count,
        total_vram,
        gpu_model,
        endpoint,
        resolutions,
    )
        .abi_encode()
}

tangle_inference_core::run_operator!(
    OperatorConfig::load(None),
    registration_payload,
    image_gen_inference::router(),
    |config| ImageGenServer { config },
    qos
);
