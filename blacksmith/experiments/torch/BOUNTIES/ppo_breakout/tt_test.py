# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr
from tt_model import BreakoutCNN

CONV1 = 0
CONV2 = 2
CONV3 = 4
FC1 = 7


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:

    if tensor1 is None or tensor2 is None:
        return float("nan")

    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape for PCC calculation.")

    if tensor1.numel() == 1 and tensor2.numel() == 1:
        return 0.0

    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()

    mean_tensor1 = tensor1_flat.mean()
    mean_tensor2 = tensor2_flat.mean()

    tensor1_zero_mean = tensor1_flat - mean_tensor1
    tensor2_zero_mean = tensor2_flat - mean_tensor2

    numerator = torch.sum(tensor1_zero_mean * tensor2_zero_mean)

    denom = torch.sqrt(torch.sum(tensor1_zero_mean**2)) * torch.sqrt(torch.sum(tensor2_zero_mean**2))

    if denom == 0:
        return float("nan")

    pcc = numerator / denom
    return pcc.item()


if __name__ == "__main__":
    xr.set_device_type("TT")
    device = torch_xla.device()
    print(f"Using device: {device}")

    agent_tt = BreakoutCNN(4, 4).to(device)
    agent_cpu = BreakoutCNN(4, 4)
    agent_tt.load_state_dict(agent_cpu.state_dict())
    # agent_cpu.load_state_dict({k: v.cpu() for k, v in agent_tt.state_dict().items()})

    # optimizers
    criterion = nn.MSELoss()
    optimizer_cpu = torch.optim.Adam(agent_cpu.parameters(), lr=2.5e-4, eps=1e-5)
    optimizer_tt = torch.optim.Adam(agent_tt.parameters(), lr=2.5e-4, eps=1e-5)

    target = torch.randn(8, 512)
    x = torch.randn(8, 4, 84, 84)  # conv1 input
    # x = torch.randn(8, 32, 20, 20) # conv2 input
    # x = torch.randn(8, 64, 9, 9) # conv3 input
    # x = torch.randn(512, 64 * 7 * 7) # flatten fc1 input

    out_cpu = agent_cpu.get_value(x)
    loss_cpu = criterion(out_cpu, target)
    optimizer_cpu.zero_grad()
    loss_cpu.backward()
    grad_cpu_conv1 = agent_cpu.network[CONV1].weight.grad.flatten()
    grad_cpu_conv2 = agent_cpu.network[CONV2].weight.grad.flatten()
    grad_cpu_conv3 = agent_cpu.network[CONV3].weight.grad.flatten()
    grad_cpu_fc1 = agent_cpu.network[FC1].weight.grad.flatten()
    grad_cpu_fc2 = agent_cpu.critic.weight.grad.flatten()
    # print(out_cpu.shape)

    x = x.to(device).requires_grad_(True)
    target = target.to(device)

    out_tt = agent_tt.get_value(x)
    torch_xla.sync(wait=True)

    loss_tt = criterion(out_tt, target)
    optimizer_tt.zero_grad()
    loss_tt.backward()
    torch_xla.sync(wait=True)
    grad_tt_conv1 = agent_tt.network[CONV1].weight.grad.flatten().cpu()
    grad_tt_conv2 = agent_tt.network[CONV2].weight.grad.flatten().cpu()
    grad_tt_conv3 = agent_tt.network[CONV3].weight.grad.flatten().cpu()
    grad_tt_fc1 = agent_tt.network[FC1].weight.grad.flatten().cpu()
    grad_tt_fc2 = agent_tt.critic.weight.grad.flatten().cpu()

    pcc_value_conv1 = compute_pcc(grad_cpu_conv1, grad_tt_conv1)
    print(f"PCC for conv1: {pcc_value_conv1}")
    pcc_value_conv2 = compute_pcc(grad_cpu_conv2, grad_tt_conv2)
    print(f"PCC for conv2: {pcc_value_conv2}")
    pcc_value_conv3 = compute_pcc(grad_cpu_conv3, grad_tt_conv3)
    print(f"PCC for conv3: {pcc_value_conv3}")
    pcc_value_fc1 = compute_pcc(grad_cpu_fc1, grad_tt_fc1)
    print(f"PCC for fc1: {pcc_value_fc1}")
    pcc_value_fc2 = compute_pcc(grad_cpu_fc2, grad_tt_fc2)
    print(f"PCC for fc2: {pcc_value_fc2}")

    # pcc_value = compute_pcc(out_tt, out_cpu)
    # print(f"Pearson Correlation Coefficient: {pcc_value}")
    # print(out_cpu)
    # print(out_tt)
