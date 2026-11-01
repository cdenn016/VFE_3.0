r"""Fused Triton GPU kernels. Importing this package imports nothing; each kernel module loads its backend
(``triton``) only when its route launches, and CPU tensors always run the torch routes.

Defaults differ by kernel. The KL decode cross-entropy kernel (``kl_decode.py``) is on by default
(``decode_ce_triton=True``) for the ``diagonal_chunked`` and ``full_chunked`` decoders on CUDA tensors. The
expected-likelihood kernel (``expected_likelihood.py``) always runs for CUDA tensors under
``decode_mode="expected_likelihood_chunked"``, with no opt-out. The diagonal-moments kernels (``diagonal_moments.py``,
``relational_diag_moments_kernel="triton"``) and the cue/result memory kernels (``cue_result.py``,
``block_memory_triton=True``) are opt-in."""
