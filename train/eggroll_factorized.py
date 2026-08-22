"""Factorized kernels for EGGROLL candidate evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from train.eggroll_perturbations import DenseVectorNoise, MatrixFactors


def _validate_linear_inputs(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    factors: Sequence[MatrixFactors],
    signs: Sequence[float] | torch.Tensor,
    bias: torch.Tensor | None,
    bias_noises: Sequence[DenseVectorNoise] | None,
) -> torch.Tensor:
    if inputs.ndim < 2:
        raise ValueError("inputs must have shape (tokens, in) or (candidates, ..., in)")
    if weight.ndim != 2:
        raise ValueError("weight must have shape (out, in)")

    out_features, in_features = weight.shape
    if inputs.shape[-1] != in_features:
        raise ValueError("input feature count does not match weight")
    if inputs.device != weight.device:
        raise ValueError("inputs and weight must use the same device")

    candidate_count = len(factors)
    if candidate_count < 1:
        raise ValueError("at least one candidate factor set is required")
    if inputs.ndim > 2 and inputs.shape[0] != candidate_count:
        raise ValueError("candidate input count does not match factor count")

    rank = factors[0].a.shape[-1] if factors[0].a.ndim == 2 else -1
    for factor in factors:
        if factor.a.shape != (out_features, rank):
            raise ValueError("matrix A shape is incompatible with weight")
        if factor.b.shape != (in_features, rank):
            raise ValueError("matrix B shape is incompatible with weight")
        if factor.a.device != inputs.device or factor.b.device != inputs.device:
            raise ValueError("matrix factors and inputs must use the same device")

    sign_tensor = torch.as_tensor(signs, device=inputs.device, dtype=inputs.dtype)
    if sign_tensor.ndim != 1 or sign_tensor.numel() != candidate_count:
        raise ValueError("sign count does not match factor count")

    if bias is not None:
        if bias.shape != (out_features,):
            raise ValueError("bias shape is incompatible with weight")
        if bias.device != inputs.device:
            raise ValueError("bias and inputs must use the same device")

    if bias_noises is not None:
        if len(bias_noises) != candidate_count:
            raise ValueError("bias-noise count does not match factor count")
        for noise in bias_noises:
            if noise.noise.shape != (out_features,):
                raise ValueError("bias-noise shape is incompatible with weight")
            if noise.noise.device != inputs.device:
                raise ValueError("bias noise and inputs must use the same device")

    return sign_tensor


def factorized_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    factors: Sequence[MatrixFactors],
    signs: Sequence[float] | torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    bias_noises: Sequence[DenseVectorNoise] | None = None,
) -> torch.Tensor:
    """Evaluate candidate linear layers without materializing perturbed weights."""
    sign_tensor = _validate_linear_inputs(
        inputs,
        weight,
        factors,
        signs,
        bias,
        bias_noises,
    )
    candidate_count = len(factors)
    stacked_a = torch.stack([factor.a for factor in factors])
    stacked_b = torch.stack([factor.b for factor in factors])
    coefficients = sign_tensor * torch.tensor(
        [factor.scale for factor in factors],
        device=inputs.device,
        dtype=inputs.dtype,
    )

    if inputs.ndim == 2:
        base = F.linear(inputs, weight, bias).unsqueeze(0).expand(
            candidate_count, -1, -1
        )
        compressed = torch.einsum("...i,cir->c...r", inputs, stacked_b)
    else:
        base = F.linear(inputs, weight, bias)
        compressed = torch.einsum("c...i,cir->c...r", inputs, stacked_b)

    residual = torch.einsum("c...r,cor->c...o", compressed, stacked_a)
    coefficient_shape = (candidate_count,) + (1,) * (residual.ndim - 1)
    output = base + residual * coefficients.reshape(coefficient_shape)

    if bias_noises is not None:
        stacked_noise = torch.stack([noise.noise for noise in bias_noises])
        noise_scales = sign_tensor * torch.tensor(
            [noise.scale for noise in bias_noises],
            device=inputs.device,
            dtype=inputs.dtype,
        )
        signed_noise = stacked_noise * noise_scales.unsqueeze(-1)
        bias_shape = (candidate_count,) + (1,) * (output.ndim - 2) + (-1,)
        output = output + signed_noise.reshape(bias_shape)

    return output
