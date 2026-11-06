// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {RegisterBlueprint} from "tangle-inference-core/script/RegisterBlueprint.s.sol";
import {console2} from "forge-std/Script.sol";
import {ImageGenBSM} from "../src/ImageGenBSM.sol";

contract Deploy is RegisterBlueprint {
    function _deployManager() internal override returns (address) {
        return address(new ImageGenBSM());
    }

    function _report(address manager, uint64 blueprintId) internal override {
        console2.log("DEPLOY_IMAGE_GEN_BSM=%s", vm.toString(manager));
        console2.log("DEPLOY_IMAGE_GEN_BLUEPRINT_ID=%s", vm.toString(blueprintId));
    }
}
