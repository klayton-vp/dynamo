# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe unit coverage for GMS torch module tensor helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is required")

try:
    from gpu_memory_service.client.torch.module import _iter_module_tensors
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _CudaLikeTensor:
    is_cuda = True


class _ModuleWithReadOnlyTensorProperty(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extra = _CudaLikeTensor()

    @property
    def expert_map(self):
        return _CudaLikeTensor()


def test_iter_module_tensors_skips_read_only_tensor_properties(monkeypatch):
    """Read-only properties should not be registered as materializable attrs."""
    monkeypatch.setattr(
        torch, "is_tensor", lambda value: isinstance(value, _CudaLikeTensor)
    )

    tensors = list(_iter_module_tensors(_ModuleWithReadOnlyTensorProperty()))

    assert [(name, tensor_type) for name, _, tensor_type in tensors] == [
        ("extra", "tensor_attr")
    ]
