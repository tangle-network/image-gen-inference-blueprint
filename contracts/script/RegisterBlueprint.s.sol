// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Script, console2 } from "forge-std/Script.sol";
import { Types } from "tnt-core/libraries/Types.sol";
import { ImageGenBSM } from "../src/ImageGenBSM.sol";

/// @notice Minimal interface for Tangle blueprint registration.
interface ITangle {
    function createBlueprint(Types.BlueprintDefinition calldata def) external returns (uint64);
}

/// @title RegisterBlueprint
/// @notice Deploys ImageGenBSM and registers the image-gen blueprint on Tangle
///         in a single broadcast.
/// @dev    ImageGenBSM is a regular (non-upgradeable) BlueprintServiceManagerBase
///         contract, so it is deployed directly without an ERC1967Proxy.
///         Run via: `forge script contracts/script/RegisterBlueprint.s.sol
///         --rpc-url $RPC_URL --broadcast --slow`
contract RegisterBlueprint is Script {
    // ─────────────────────────────────────────────────────────────────────────
    // Defaults — overridable via env vars for non-anvil chains.
    // ─────────────────────────────────────────────────────────────────────────

    // Anvil well-known deployer key (default when no PRIVATE_KEY env is set).
    uint256 constant DEFAULT_DEPLOYER_KEY =
        0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;

    // Tangle protocol address on a LocalTestnet anvil snapshot. For real
    // chains (Base Sepolia, mainnet) pass TANGLE_CORE via env.
    address constant DEFAULT_TANGLE = 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9;

    // USDC on Base Sepolia. Per-image billing settles in this token. For other
    // networks pass PAYMENT_TOKEN via env. The address is captured here purely
    // for visibility in deployment logs — ImageGenBSM does not consume it at
    // construction time. The blueprint owner configures pricing post-deploy
    // via `ImageGenBSM.configureModel`.
    address constant DEFAULT_PAYMENT_TOKEN = 0x036CbD53842c5426634e7929541eC2318f3dCF7e;

    function run() external {
        uint256 deployerKey = vm.envOr("PRIVATE_KEY", DEFAULT_DEPLOYER_KEY);
        address tangleAddr = vm.envOr("TANGLE_CORE", DEFAULT_TANGLE);
        address paymentToken = vm.envOr("PAYMENT_TOKEN", DEFAULT_PAYMENT_TOKEN);

        ITangle tangle = ITangle(tangleAddr);

        vm.startBroadcast(deployerKey);

        // ── Deploy ImageGenBSM (non-upgradeable, no constructor args) ───────
        ImageGenBSM bsm = new ImageGenBSM();

        // ── Register on Tangle ──────────────────────────────────────────────
        uint64 blueprintId = tangle.createBlueprint(_buildDefinition(address(bsm)));

        vm.stopBroadcast();

        // ── Output for bash wrapper parsing ─────────────────────────────────
        console2.log("DEPLOY_IMAGE_GEN_BSM=%s", vm.toString(address(bsm)));
        console2.log("DEPLOY_IMAGE_GEN_BLUEPRINT_ID=%s", vm.toString(blueprintId));
        console2.log("DEPLOY_IMAGE_GEN_PAYMENT_TOKEN=%s", vm.toString(paymentToken));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Blueprint Definition builder
    // ═════════════════════════════════════════════════════════════════════════

    function _buildDefinition(address manager) internal pure returns (Types.BlueprintDefinition memory def) {
        def.metadataUri = "https://github.com/tangle-network/image-gen-inference-blueprint";
        // metadataHash is a digest of the canonical metadata JSON. Until that
        // payload is pinned via IPFS, derive it from the metadataUri so the
        // value is deterministic + traceable.
        def.metadataHash = keccak256(bytes(def.metadataUri));
        def.manager = manager;
        def.masterManagerRevision = 0;
        def.hasConfig = true;

        // Event-driven pricing: operators are paid per image generated rather
        // than on a fixed subscription cadence. Dynamic membership lets new
        // GPU operators join after registration.
        def.config = Types.BlueprintConfig({
            membership: Types.MembershipModel.Dynamic,
            pricing: Types.PricingModel.EventDriven,
            minOperators: 1,
            maxOperators: 0, // unbounded
            subscriptionRate: 0,
            subscriptionInterval: 0,
            eventRate: 0 // operators set price per image via configureModel
         });

        def.metadata = Types.BlueprintMetadata({
            name: "Image Generation Blueprint",
            description: "Diffusion-backed image generation operator (Stable Diffusion, FLUX, SDXL via ComfyUI/diffusers)",
            author: "Tangle",
            category: "AI/Inference",
            codeRepository: "https://github.com/tangle-network/image-gen-inference-blueprint",
            logo: "",
            website: "https://tangle.network",
            license: "MIT",
            profilingData: ""
        });

        def.jobs = _buildJobs();

        def.registrationSchema = "";
        def.requestSchema = "";

        def.sources = new Types.BlueprintSource[](1);
        Types.BlueprintBinary[] memory bins = new Types.BlueprintBinary[](1);
        bins[0] = Types.BlueprintBinary({
            arch: Types.BlueprintArchitecture.Amd64,
            os: Types.BlueprintOperatingSystem.Linux,
            name: "image-gen-inference-blueprint",
            sha256: bytes32(uint256(0xdeadbeef))
        });
        def.sources[0] = Types.BlueprintSource({
            kind: Types.BlueprintSourceKind.Native,
            container: Types.ImageRegistrySource("", "", ""),
            wasm: Types.WasmSource(Types.WasmRuntime.Unknown, Types.BlueprintFetcherKind.None, "", ""),
            native: Types.NativeSource(
                Types.BlueprintFetcherKind.None,
                "file:///target/release/image-gen-inference-blueprint",
                "./target/release/image-gen-inference-blueprint"
            ),
            testing: Types.TestingSource("image-gen-inference-blueprint-bin", "image-gen-inference-blueprint", "."),
            binaries: bins
        });

        def.supportedMemberships = new Types.MembershipModel[](1);
        def.supportedMemberships[0] = Types.MembershipModel.Dynamic;
    }

    function _buildJobs() internal pure returns (Types.JobDefinition[] memory jobs) {
        jobs = new Types.JobDefinition[](1);
        // Job 0: image generation
        //   inputs:  (string prompt, uint32 width, uint32 height, uint32 steps, uint32 numImages)
        //   outputs: (string imageUri, uint32 numImages, uint32 widthUsed, uint32 heightUsed)
        // The Rust operator (operator/src/lib.rs `run_image_gen`) enforces the
        // ABI; on-chain schemas stay empty here to match the pattern used by
        // the sibling vLLM inference blueprint.
        jobs[0] = Types.JobDefinition({
            name: "image-gen",
            description: "Generate image(s) from a prompt via the operator's diffusion backend",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });
    }
}
