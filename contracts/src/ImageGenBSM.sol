// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @title ImageGenBSM — Blueprint Service Manager for image generation
/// @dev Fork of InferenceBSM with per-image pricing instead of per-token
/// Operators register with GPU capabilities and serve image generation models
/// (Stable Diffusion, FLUX, SDXL, etc.) via ComfyUI or diffusers endpoints.
///
/// Pricing: pricePerImage (flat rate per generation)
/// Model validation: minGpuVramMib based on model requirements
/// Registration: model, gpuCount, totalVramMib, gpuModel, endpoint

import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

contract ImageGenBSM is BlueprintServiceManagerBase {
    struct OperatorCapabilities {
        string model;
        uint32 gpuCount;
        uint32 totalVramMib;
        string gpuModel;
        string endpoint;
        bool active;
    }

    struct ModelConfig {
        uint64 pricePerImage;
        uint32 minGpuVramMib;
        bool enabled;
    }

    mapping(address => OperatorCapabilities) public operatorCaps;
    mapping(bytes32 => ModelConfig) public modelConfigs;
    address[] internal _operators;

    function configureModel(string calldata model, uint64 pricePerImage, uint32 minGpuVramMib) external onlyBlueprintOwner {
        modelConfigs[keccak256(bytes(model))] = ModelConfig(pricePerImage, minGpuVramMib, true);
    }

    function onRegister(address operator, bytes calldata registrationInputs) external payable override onlyFromTangle {
        (string memory model, uint32 gpuCount, uint32 totalVramMib, string memory gpuModel, string memory endpoint)
            = abi.decode(registrationInputs, (string, uint32, uint32, string, string));
        ModelConfig memory cfg = modelConfigs[keccak256(bytes(model))];
        require(cfg.enabled, "Model not supported");
        require(totalVramMib >= cfg.minGpuVramMib, "Insufficient VRAM");
        operatorCaps[operator] = OperatorCapabilities(model, gpuCount, totalVramMib, gpuModel, endpoint, true);
        _operators.push(operator);
    }

    function getOperators() external view returns (address[] memory) { return _operators; }
    function isOperatorActive(address op) external view returns (bool) { return operatorCaps[op].active; }
    function getOperatorPricing(address op) external view returns (uint64 pricePerImage, string memory endpoint) {
        OperatorCapabilities memory c = operatorCaps[op];
        ModelConfig memory cfg = modelConfigs[keccak256(bytes(c.model))];
        return (cfg.pricePerImage, c.endpoint);
    }
    function onUnregister(address operator) external override onlyFromTangle { operatorCaps[operator].active = false; }
    function onUpdatePreferences(address operator, bytes calldata p) external payable override onlyFromTangle {
        operatorCaps[operator].endpoint = abi.decode(p, (string));
    }
}
