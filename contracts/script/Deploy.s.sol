// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";
import {ImageGenBSM} from "../src/ImageGenBSM.sol";

/// Model-specific manager deployment; cargo-tangle registers the definition.
contract Deploy is Script {
    function run() external {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        vm.startBroadcast();
        address manager = address(new ImageGenBSM(registry));
        vm.stopBroadcast();
        console2.log("DEPLOY_IMAGE_GEN_BSM=%s", vm.toString(manager));
    }
}
