// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Test } from "forge-std/Test.sol";
import { ImageGenBSM } from "../src/ImageGenBSM.sol";
import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

contract ImageGenBSMTest is Test {
    ImageGenBSM public bsm;

    address public tangleCore = address(0xC0DE);
    address public owner = address(0xBEEF);
    address public operator1 = address(0x1111);
    address public operator2 = address(0x2222);
    address public user = address(0x3333);

    function setUp() public {
        bsm = new ImageGenBSM();
        bsm.onBlueprintCreated(1, owner, tangleCore);

        vm.prank(owner);
        bsm.configureModel("stable-diffusion-xl", 500_000, 16_000); // 0.50 USD per image, 16GB VRAM
    }

    // --- Initialization ---

    function test_initialization() public view {
        assertEq(bsm.blueprintId(), 1);
        assertEq(bsm.blueprintOwner(), owner);
        assertEq(bsm.tangleCore(), tangleCore);
    }

    function test_cannotReinitialize() public {
        vm.expectRevert(BlueprintServiceManagerBase.AlreadyInitialized.selector);
        bsm.onBlueprintCreated(2, owner, tangleCore);
    }

    // --- Model Configuration ---

    function test_configureModel() public {
        vm.prank(owner);
        bsm.configureModel("flux-1-dev", 1_000_000, 24_000);

        (uint64 price, uint32 minVram, bool enabled) = bsm.modelConfigs(keccak256("flux-1-dev"));
        assertEq(price, 1_000_000);
        assertEq(minVram, 24_000);
        assertTrue(enabled);
    }

    function test_configureModel_onlyOwner() public {
        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyBlueprintOwnerAllowed.selector, user, owner)
        );
        bsm.configureModel("test-model", 100, 8000);
    }

    // --- Operator Registration ---

    function test_registerOperator() public {
        bytes memory regData = abi.encode(
            "stable-diffusion-xl",
            uint32(2),
            uint32(48_000),
            "NVIDIA A100",
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        bsm.onRegister(operator1, regData);

        assertTrue(bsm.isOperatorActive(operator1));

        address[] memory ops = bsm.getOperators();
        assertEq(ops.length, 1);
        assertEq(ops[0], operator1);
    }

    function test_registerOperator_unsupportedModel() public {
        bytes memory regData = abi.encode(
            "nonexistent-model",
            uint32(1),
            uint32(16_000),
            "NVIDIA A100",
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        vm.expectRevert("Model not supported");
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_insufficientVram() public {
        bytes memory regData = abi.encode(
            "stable-diffusion-xl",
            uint32(1),
            uint32(8_000), // below 16_000 minimum
            "NVIDIA RTX 3060",
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        vm.expectRevert("Insufficient VRAM");
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_onlyTangle() public {
        bytes memory regData = abi.encode(
            "stable-diffusion-xl",
            uint32(1),
            uint32(16_000),
            "NVIDIA A100",
            "https://op1.example.com"
        );

        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyTangleAllowed.selector, user, tangleCore)
        );
        bsm.onRegister(operator1, regData);
    }

    // --- Pricing ---

    function test_getOperatorPricing() public {
        _registerOperator(operator1);

        (uint64 pricePerImage, string memory endpoint) = bsm.getOperatorPricing(operator1);
        assertEq(pricePerImage, 500_000);
        assertEq(keccak256(bytes(endpoint)), keccak256(bytes("https://op1.example.com")));
    }

    // --- Unregistration ---

    function test_unregisterOperator() public {
        _registerOperator(operator1);

        vm.prank(tangleCore);
        bsm.onUnregister(operator1);

        assertFalse(bsm.isOperatorActive(operator1));
    }

    // --- Update Preferences ---

    function test_updatePreferences() public {
        _registerOperator(operator1);

        vm.prank(tangleCore);
        bsm.onUpdatePreferences(operator1, abi.encode("https://new-endpoint.example.com"));

        (, string memory endpoint) = bsm.getOperatorPricing(operator1);
        assertEq(keccak256(bytes(endpoint)), keccak256(bytes("https://new-endpoint.example.com")));
    }

    // --- Multiple Operators ---

    function test_multipleOperators() public {
        _registerOperator(operator1);

        bytes memory regData2 = abi.encode(
            "stable-diffusion-xl",
            uint32(4),
            uint32(80_000),
            "NVIDIA H100",
            "https://op2.example.com"
        );
        vm.prank(tangleCore);
        bsm.onRegister(operator2, regData2);

        address[] memory ops = bsm.getOperators();
        assertEq(ops.length, 2);
    }

    // --- Helpers ---

    function _registerOperator(address op) internal {
        bytes memory regData = abi.encode(
            "stable-diffusion-xl",
            uint32(2),
            uint32(48_000),
            "NVIDIA A100",
            "https://op1.example.com"
        );
        vm.prank(tangleCore);
        bsm.onRegister(op, regData);
    }
}
