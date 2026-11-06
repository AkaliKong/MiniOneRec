"""Ascend NPU enablement shim for MiniOneRec (backward-compatible).

MiniOneRec is written against the CUDA-oriented ``torch.cuda`` API
(``torch.cuda.is_available``, ``device_count``, ``manual_seed``, the
``"cuda"`` device string, ``CUDA_VISIBLE_DEVICES``, ...). On Ascend NPU
machines the ``torch_npu`` package ships a drop-in remapping
(``torch_npu.contrib.transfer_to_npu``) that transparently redirects the
entire ``torch.cuda`` surface onto the NPU backend, so the existing code
runs unchanged on NPU.

This module activates that remapping *only* when ``torch_npu`` is
installed. On CUDA / CPU builds ``torch_npu`` is absent and the import is
skipped, leaving every original behaviour intact -- there is no change to
model logic, training, or evaluation on existing CUDA hardware.

Usage: add ``import npu_support`` as the *first* import in any runnable
entry script. Nothing else in the codebase needs to change, except where a
line picks a backend-specific library (e.g. a bitsandbytes
optimizer), in which case branch on ``npu_support.IS_NPU`` instead of
``torch.cuda.is_available()`` -- the remapping above makes the latter return
True on NPU as well.
"""

try:
    import torch_npu  # noqa: F401  (registers the NPU backend)
    from torch_npu.contrib import transfer_to_npu  # noqa: F401  (monkey-patches torch.cuda -> npu)
    IS_NPU = True
except ImportError:
    # Ascend NPU stack not installed (CUDA / CPU build). No-op on purpose.
    IS_NPU = False
