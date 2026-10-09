# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import torch

from shardstream import logging

logger = logging.getLogger(__name__)


def check_and_log_nan_values(tensor, tensor_name, stage_info="", max_indices=20):
    """
    Check for NaN values in a tensor and log detailed information including indices.

    Args:
        tensor: The tensor to check for NaN values
        tensor_name: Name of the tensor for logging
        stage_info: Additional stage information for logging context
        max_indices: Maximum number of NaN indices to log (default: 20)

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        logger.warning(
            f"Parameter {tensor_name} contains NaN values{stage_info}! Shape: {tensor.shape}"
        )
        logger.warning(f"NaN count in {tensor_name}: {nan_count}")
        nan_indices = torch.nonzero(torch.isnan(tensor), as_tuple=True)
        logger.warning(f"NaN indices in {tensor_name}{stage_info}: {nan_indices}")
        if nan_count <= max_indices:
            logger.warning(
                f"All NaN values in {tensor_name}{stage_info}: {nan_indices}"
            )
        else:
            logger.warning(
                f"First {max_indices} NaN indices in {tensor_name}{stage_info}: {tuple((idx[:max_indices] for idx in nan_indices))}"
            )
        return True
    return False


def compare_and_log_tensor_differences(
    tensor1,
    tensor2,
    tensor_name,
    atol=1e-08,
    rtol=1e-05,
    max_differences=20,
    exact_match=False,
):
    """
    Compare two tensors and log detailed information about inconsistent elements.

    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        tensor_name: Name of the tensor for logging
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        max_differences: Maximum number of differences to log (default: 20)

    Returns:
        bool: True if tensors are consistent, False otherwise
    """
    if tensor1.shape != tensor2.shape:
        logger.error(
            f"Shape mismatch for {tensor_name}: {tensor1.shape} vs {tensor2.shape}"
        )
        return False
    if tensor1.dtype != tensor2.dtype:
        logger.error(
            f"Tensor {tensor_name} has different dtypes: {tensor1.dtype} vs {tensor2.dtype}"
        )
        return False
    if exact_match:
        if torch.equal(tensor1, tensor2):
            return True
    elif torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
        return True
    logger.error(
        f"Tensors are not close for {tensor_name}, get {tensor1.shape} \n{tensor1} expect {tensor2.shape} \n{tensor2}"
    )
    close_mask = torch.isclose(tensor1, tensor2, atol=atol, rtol=rtol)
    inconsistent_mask = ~close_mask
    if inconsistent_mask.any():
        inconsistent_count = inconsistent_mask.sum().item()
        logger.error(
            f"Parameter {tensor_name} has {inconsistent_count} inconsistent elements"
        )
        inconsistent_indices = torch.nonzero(inconsistent_mask, as_tuple=True)
        abs_diff = torch.abs(tensor1 - tensor2)
        rel_diff = abs_diff / (torch.abs(tensor2) + 1e-08)
        tensor1_values = tensor1[inconsistent_indices]
        tensor2_values = tensor2[inconsistent_indices]
        abs_diff_values = abs_diff[inconsistent_indices]
        rel_diff_values = rel_diff[inconsistent_indices]
        max_abs_diff = abs_diff_values.max().item()
        max_rel_diff = rel_diff_values.max().item()
        mean_abs_diff = abs_diff_values.mean().item()
        mean_rel_diff = rel_diff_values.mean().item()
        logger.error(
            f"Max absolute difference: {max_abs_diff:.6f}, Max relative difference: {max_rel_diff:.6f}"
        )
        logger.error(
            f"Mean absolute difference: {mean_abs_diff:.6f}, Mean relative difference: {mean_rel_diff:.6f}"
        )
        num_to_log = min(int(inconsistent_count), max_differences)
        for i in range(num_to_log):
            idx = tuple((idx[i] for idx in inconsistent_indices))
            val1 = tensor1_values[i].item()
            val2 = tensor2_values[i].item()
            abs_diff_val = abs_diff_values[i].item()
            rel_diff_val = rel_diff_values[i].item()
            logger.error(
                f"  Index {idx}: {val1:.6f} vs {val2:.6f} (abs_diff: {abs_diff_val:.6f}, rel_diff: {rel_diff_val:.6f})"
            )
        if inconsistent_count > max_differences:
            logger.error(
                f"  ... and {inconsistent_count - max_differences} more differences"
            )
        return False
    return True
