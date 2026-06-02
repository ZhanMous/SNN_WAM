"""DWM-G2: Transition dataset no-future-leakage tests.

Verifies:
- Patch latent transition dataset has correct shapes
- No future information leaks into context
- Causal alignment is correct (context at t, target at t+1)
- Synthetic anti-leakage test with time-encoded values
- Cached patch latent dataset loads correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from src.data.patch_latent_dataset import (
    PatchLatentTransitionDataset,
    create_dinowm_transition_dataset,
    load_patch_latent_cache,
)
from src.data.trajectory_window import (
    RawTrajectory,
    TrajectoryWindowDataset,
    make_mock_trajectory_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
PATCH_LATENT_DIR = ROOT / "latents" / "libero_spatial" / "dinov2_vits14_patch"


# ---------------------------------------------------------------------------
# DWM-G2 Gate 2: Transition dataset shape tests
# ---------------------------------------------------------------------------


class TestDWMG2TransitionShapes:
    """Verify transition dataset has correct shapes."""

    def test_mock_dataset_shapes(self) -> None:
        """Mock dataset has correct shapes for z_context, actions, z_target."""
        ds = create_dinowm_transition_dataset(
            cache_dir=PATCH_LATENT_DIR,
            context_len=3,
            future_horizon=2,
            max_demos=1,
            max_frames=20,
        )

        if len(ds) == 0:
            pytest.skip("No data loaded")

        sample = ds[0]

        # z_context should be [context_len, P, D]
        assert "z_context" in sample
        z_ctx = sample["z_context"]
        assert z_ctx.ndim == 3, f"z_context should be 3D, got {z_ctx.ndim}D"
        assert z_ctx.shape[0] == 3, f"Expected context_len=3, got {z_ctx.shape[0]}"

        # actions should be [context_len, A]
        assert "actions" in sample
        actions = sample["actions"]
        assert actions.ndim == 2, f"actions should be 2D, got {actions.ndim}D"
        assert actions.shape[0] == 3, f"Expected context_len=3, got {actions.shape[0]}"

        # z_target should be [future_horizon, P, D]
        assert "z_target" in sample
        z_tgt = sample["z_target"]
        assert z_tgt.ndim == 3, f"z_target should be 3D, got {z_tgt.ndim}D"
        assert z_tgt.shape[0] == 2, f"Expected future_horizon=2, got {z_tgt.shape[0]}"

        # P and D should match between context and target
        assert z_ctx.shape[1] == z_tgt.shape[1], "Num patches mismatch"
        assert z_ctx.shape[2] == z_tgt.shape[2], "Feature dim mismatch"

    def test_patch_dim_property(self) -> None:
        """patch_dim returns (num_patches, feature_dim)."""
        ds = create_dinowm_transition_dataset(
            cache_dir=PATCH_LATENT_DIR,
            context_len=3,
            future_horizon=2,
            max_demos=1,
            max_frames=20,
        )

        if len(ds) == 0:
            pytest.skip("No data loaded")

        P, D = ds.patch_dim
        assert P > 0, f"Expected positive num_patches, got {P}"
        assert D > 0, f"Expected positive feature_dim, got {D}"

    def test_action_dim_property(self) -> None:
        """action_dim returns positive integer."""
        ds = create_dinowm_transition_dataset(
            cache_dir=PATCH_LATENT_DIR,
            context_len=3,
            future_horizon=2,
            max_demos=1,
            max_frames=20,
        )

        if len(ds) == 0:
            pytest.skip("No data loaded")

        A = ds.action_dim
        assert A > 0, f"Expected positive action_dim, got {A}"


# ---------------------------------------------------------------------------
# DWM-G2 Gate 2: No-future-leakage tests
# ---------------------------------------------------------------------------


class TestDWMG2NoFutureLeakage:
    """Verify no future information leaks into context."""

    def test_context_does_not_contain_target(self) -> None:
        """Context patch latents are from before target timesteps."""
        ds = create_dinowm_transition_dataset(
            cache_dir=PATCH_LATENT_DIR,
            context_len=3,
            future_horizon=2,
            max_demos=1,
            max_frames=20,
        )

        if len(ds) == 0:
            pytest.skip("No data loaded")

        sample = ds[0]
        metadata = sample["metadata"]

        ctx_range = metadata["context_range"]
        tgt_range = metadata["target_range"]

        # Context and target ranges should not overlap
        assert not set(ctx_range) & set(tgt_range), (
            f"Context {ctx_range} and target {tgt_range} overlap"
        )

        # Context should be before target
        assert max(ctx_range) < min(tgt_range), (
            f"Context max {max(ctx_range)} >= target min {min(tgt_range)}"
        )

    def test_context_indices_are_contiguous(self) -> None:
        """Context indices are contiguous and in order."""
        ds = create_dinowm_transition_dataset(
            cache_dir=PATCH_LATENT_DIR,
            context_len=3,
            future_horizon=2,
            max_demos=1,
            max_frames=20,
        )

        if len(ds) == 0:
            pytest.skip("No data loaded")

        sample = ds[0]
        ctx_range = sample["metadata"]["context_range"]

        # Should be contiguous
        for i in range(len(ctx_range) - 1):
            assert ctx_range[i + 1] == ctx_range[i] + 1, (
                f"Context not contiguous: {ctx_range}"
            )

    def test_target_indices_are_contiguous(self) -> None:
        """Target indices are contiguous and in order."""
        ds = create_dinowm_transition_dataset(
            cache_dir=PATCH_LATENT_DIR,
            context_len=3,
            future_horizon=2,
            max_demos=1,
            max_frames=20,
        )

        if len(ds) == 0:
            pytest.skip("No data loaded")

        sample = ds[0]
        tgt_range = sample["metadata"]["target_range"]

        # Should be contiguous
        for i in range(len(tgt_range) - 1):
            assert tgt_range[i + 1] == tgt_range[i] + 1, (
                f"Target not contiguous: {tgt_range}"
            )

    def test_target_starts_after_context(self) -> None:
        """Target starts immediately after context ends."""
        ds = create_dinowm_transition_dataset(
            cache_dir=PATCH_LATENT_DIR,
            context_len=3,
            future_horizon=2,
            max_demos=1,
            max_frames=20,
        )

        if len(ds) == 0:
            pytest.skip("No data loaded")

        sample = ds[0]
        ctx_range = sample["metadata"]["context_range"]
        tgt_range = sample["metadata"]["target_range"]

        # Target should start right after context
        assert tgt_range[0] == ctx_range[-1] + 1, (
            f"Target starts at {tgt_range[0]}, expected {ctx_range[-1] + 1}"
        )


# ---------------------------------------------------------------------------
# DWM-G2 Gate 2: Synthetic anti-leakage test
# ---------------------------------------------------------------------------


class TestDWMG2SyntheticAntiLeakage:
    """Synthetic test with time-encoded values to detect leakage."""

    def test_time_encoded_no_leakage(self) -> None:
        """Each timestep has unique values; verify no mixing."""
        # Create synthetic trajectory with time-encoded patch latents
        T, P, D, A = 10, 4, 2, 3
        context_len, future_horizon = 3, 2

        # Create time-encoded patch latents
        patch_latents = []
        for t in range(T):
            # Each timestep has unique value: t + patch_offset + feature_offset
            latents = [
                [float(t) + 0.1 * p + 0.01 * d for d in range(D)]
                for p in range(P)
            ]
            patch_latents.append(latents)

        # Create time-encoded actions
        actions = [[float(t) for _ in range(A)] for t in range(T)]

        trajectory = RawTrajectory(
            images=list(range(T)),
            actions=actions,
            language="synthetic test",
            patch_latents=patch_latents,
            trajectory_id="synthetic_0",
            split="train",
        )

        ds = TrajectoryWindowDataset(
            [trajectory],
            history_len=context_len,
            action_horizon=context_len,
            future_horizon=future_horizon,
            include_current_patch_latents=True,
            include_future_patch_latents=True,
            split="train",
        )

        # Find a sample at time t=5
        for i in range(len(ds)):
            sample = ds[i]
            t = sample["time_index"]
            if t == 5:
                # Context should be patch_latents[3:6] (t=3,4,5)
                z_ctx = torch.tensor(sample["z_t_patch"])
                # z_ctx shape is [P, D] (single timestep), not [context_len, P, D]
                # The dataset returns single timestep patch latents
                assert z_ctx.shape == (P, D), f"Expected shape ({P}, {D}), got {z_ctx.shape}"

                # Verify context values match expected t=5
                expected_t = 5
                expected_vals = torch.tensor([
                    [float(expected_t) + 0.1 * p + 0.01 * d for d in range(D)]
                    for p in range(P)
                ])
                assert torch.allclose(z_ctx, expected_vals, atol=1e-5), (
                    f"Context values: expected t={expected_t}, got {z_ctx}"
                )

                # Target should be patch_latents[6:8] (t=6,7)
                z_tgt = torch.tensor(sample["target_future_patch_latents"])
                assert z_tgt.shape == (future_horizon, P, D)

                for h in range(future_horizon):
                    expected_t = 6 + h  # t=6,7
                    actual_vals = z_tgt[h]
                    expected_vals = torch.tensor([
                        [float(expected_t) + 0.1 * p + 0.01 * d for d in range(D)]
                        for p in range(P)
                    ])
                    assert torch.allclose(actual_vals, expected_vals, atol=1e-5), (
                        f"Target step {h}: expected t={expected_t}, got values {actual_vals}"
                    )

                # Verify no leakage: context values should not equal target values
                tgt_first = z_tgt[0]
                assert not torch.allclose(z_ctx, tgt_first, atol=1e-5), (
                    "Context equals target first step - potential leakage"
                )

                return  # Test passed

        pytest.skip("No sample at t=5 found")


# ---------------------------------------------------------------------------
# DWM-G2 Gate 2: Cached dataset tests
# ---------------------------------------------------------------------------


class TestDWMG2CachedDataset:
    """Verify cached patch latent dataset loads correctly."""

    def test_cache_loads(self) -> None:
        """Cached patch latents load without error."""
        if not PATCH_LATENT_DIR.exists():
            pytest.skip("No patch latent cache")

        ds = create_dinowm_transition_dataset(
            cache_dir=PATCH_LATENT_DIR,
            context_len=3,
            future_horizon=2,
            max_demos=1,
            max_frames=20,
        )

        assert len(ds) > 0, "Dataset should have samples"

    def test_cached_shapes_match(self) -> None:
        """Cached dataset has correct shapes."""
        if not PATCH_LATENT_DIR.exists():
            pytest.skip("No patch latent cache")

        ds = create_dinowm_transition_dataset(
            cache_dir=PATCH_LATENT_DIR,
            context_len=3,
            future_horizon=2,
            max_demos=1,
            max_frames=20,
        )

        if len(ds) == 0:
            pytest.skip("No samples")

        sample = ds[0]
        P, D = ds.patch_dim

        assert sample["z_context"].shape == (3, P, D)
        assert sample["z_target"].shape == (2, P, D)

    def test_multiple_samples_are_different(self) -> None:
        """Different samples have different time indices."""
        if not PATCH_LATENT_DIR.exists():
            pytest.skip("No patch latent cache")

        ds = create_dinowm_transition_dataset(
            cache_dir=PATCH_LATENT_DIR,
            context_len=3,
            future_horizon=2,
            max_demos=1,
            max_frames=30,
        )

        if len(ds) < 2:
            pytest.skip("Need at least 2 samples")

        sample1 = ds[0]
        sample2 = ds[len(ds) // 2]

        # Time indices should be different
        assert sample1["metadata"]["time_index"] != sample2["metadata"]["time_index"]
