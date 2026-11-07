// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";
import {ImageGenBSM} from "../src/ImageGenBSM.sol";

/// Model-specific manager deployment; cargo-tangle registers the definition.
contract Deploy is Script {
    function run() external {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        // Without a configured model every operator registration reverts; a zero price would serve images for free.
        uint256 price = vm.envUint("PRICE_PER_IMAGE");
        require(price > 0, "PRICE_PER_IMAGE must be positive");
        vm.startBroadcast();
        ImageGenBSM manager = new ImageGenBSM(registry);
        manager.configureModel(vm.envString("MODEL"), uint64(price), uint32(vm.envUint("MIN_GPU_VRAM_MIB")));
        vm.stopBroadcast();
        console2.log("DEPLOY_MANAGER=%s", vm.toString(address(manager)));
    }
}
