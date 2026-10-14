"""Flow-matching training objective (#115).

FlowMatchingPlanner.forward() Euler-integrates from fresh noise, so regressing
its output onto the target is NOT the flow objective and drives the planner to
the conditional mean. These tests pin the replacement:

* ``FlowMatchingPlanner.compute_planner_loss`` is the velocity-MSE objective,
  computed through the same conditioning path as ``forward()``.
* Optimising ONLY that loss trains the inference path: ``forward()``'s rollout
  converges onto the target.
* ``AutoE2E.forward(..., return_planner_loss=True)`` surfaces it as
  ``aux["planner_loss"]`` in train mode, which is what ``train_il`` consumes for
  ``planner_mode="flow_matching"``.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.append('..')

from model_components.trajectory_planning import FlowMatchingPlanner

EMBED_DIM = 256
TRAJ_DIM = 128


def _planner_inputs(batch_size, device, bev_hw=4):
    bev = torch.randn(batch_size, EMBED_DIM, bev_hw, bev_hw, device=device)
    visual_history = torch.randn(batch_size, 896, device=device)
    egomotion = torch.randn(batch_size, 256, device=device)
    return bev, visual_history, egomotion


def _model_inputs(batch_size, num_views, device):
    visual = torch.randn(batch_size, num_views, 3, 256, 256, device=device)
    map_input = torch.randn(batch_size, 3, 256, 256, device=device)
    visual_history = torch.randn(batch_size, 896, device=device)
    egomotion = torch.randn(batch_size, 256, device=device)
    return visual, map_input, visual_history, egomotion


class TestComputePlannerLoss:
    def test_returns_finite_scalar_with_gradients(self, device):
        planner = FlowMatchingPlanner(embed_dim=EMBED_DIM).to(device)
        bev, vis, ego = _planner_inputs(2, device)
        target = torch.randn(2, TRAJ_DIM, device=device)

        loss = planner.compute_planner_loss(bev, vis, ego, target)

        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert planner.velocity_head.weight.grad is not None
        assert planner.velocity_head.weight.grad.abs().sum() > 0

    def test_is_velocity_mse_through_forward_conditioning(self, device):
        """Same RNG draw => identical to MSE(v_theta(u_t, t, c), x1 - x0)
        built from the helpers forward() uses."""
        planner = FlowMatchingPlanner(embed_dim=EMBED_DIM).to(device).eval()
        bev, vis, ego = _planner_inputs(2, device)
        target = torch.randn(2, TRAJ_DIM, device=device)

        torch.manual_seed(7)
        loss = planner.compute_planner_loss(bev, vis, ego, target)

        torch.manual_seed(7)
        mod_cond = planner._modulation_conditioning(vis, ego)
        bev_seq = planner._project_bev(bev)
        u_t, t, target_velocity = planner.construct_training_data(target)
        expected = F.mse_loss(
            planner._v_theta(u_t, t, bev_seq, mod_cond), target_velocity,
        )
        torch.testing.assert_close(loss, expected)

    def test_rejects_wrong_target_shape(self, device):
        planner = FlowMatchingPlanner(embed_dim=EMBED_DIM).to(device)
        bev, vis, ego = _planner_inputs(2, device)
        with pytest.raises(ValueError, match="trajectory_target"):
            planner.compute_planner_loss(
                bev, vis, ego, torch.randn(2, TRAJ_DIM - 2, device=device),
            )

    def test_training_on_it_makes_forward_reach_a_time_varying_target(self, device):
        """Optimising only the velocity objective must make the Euler rollout
        that inference uses land on a target whose controls CHANGE over the
        horizon (as every real accel/curvature profile does).

        Regression guard for the missing horizon-position embedding: without
        it the 64 action tokens are indistinguishable and this plateaus at a
        mean rollout error of ~0.57 (≈ E|x0|, i.e. the noise is never removed);
        with it the error drops to ~0.03.
        """
        torch.manual_seed(0)
        planner = FlowMatchingPlanner(
            embed_dim=EMBED_DIM, num_inference_steps=10,
        ).to(device)
        bev, vis, ego = _planner_inputs(1, device)
        target = torch.linspace(-1.0, 1.0, TRAJ_DIM, device=device)[None]
        noise = torch.randn(1, TRAJ_DIM, device=device,
                            generator=torch.Generator(device).manual_seed(1))

        optimizer = torch.optim.Adam(planner.parameters(), lr=1e-3)
        for _ in range(500):
            optimizer.zero_grad()
            planner.compute_planner_loss(
                bev.expand(32, -1, -1, -1), vis.expand(32, -1),
                ego.expand(32, -1), target.expand(32, -1),
            ).backward()
            optimizer.step()

        planner.eval()
        with torch.no_grad():
            pred = planner(bev, vis, ego, initial_noise=noise)
        error = (pred - target).abs().mean().item()
        assert error < 0.15, error

    def test_action_tokens_are_horizon_aware(self, device):
        """Identical (accel, curv) values at different future timesteps must
        produce different queries; otherwise no time-varying plan is
        expressible."""
        planner = FlowMatchingPlanner(embed_dim=EMBED_DIM).to(device)
        same_value_everywhere = torch.zeros(1, TRAJ_DIM, device=device)
        bev, vis, ego = _planner_inputs(1, device)
        mod_cond = planner._modulation_conditioning(vis, ego)
        bev_seq = planner._project_bev(bev)
        t = torch.full((1,), 0.5, device=device)
        velocity = planner._v_theta(
            same_value_everywhere, t, bev_seq, mod_cond,
        ).reshape(1, planner.num_timesteps, planner.num_signals)
        assert not torch.allclose(velocity[:, 0], velocity[:, -1])


class TestAutoE2EReturnPlannerLoss:
    @staticmethod
    def _model(build_mock_model, device, planner_mode):
        kwargs = {"num_inference_steps": 4} if planner_mode == "flow_matching" else None
        return build_mock_model(
            num_views=8, device=device,
            planner_mode=planner_mode, planner_kwargs=kwargs,
        )

    def test_train_mode_returns_planner_loss_in_aux(self, build_mock_model, device):
        model = self._model(build_mock_model, device, "flow_matching").train()
        visual, map_input, vis, ego = _model_inputs(2, 8, device)
        target = torch.randn(2, TRAJ_DIM, device=device)

        out = model(visual, map_input, vis, ego, mode="train",
                    trajectory_target=target, return_planner_loss=True)

        trajectory, aux = out
        assert trajectory.shape == (2, TRAJ_DIM)
        loss = aux["planner_loss"]
        assert loss.dim() == 0 and torch.isfinite(loss)
        loss.backward()
        assert model.Reactive_E2E.TrajectoryPlanner.velocity_head.weight.grad is not None

    def test_default_contract_unchanged(self, build_mock_model, device):
        """Without the flag, train mode still returns the bare trajectory."""
        model = self._model(build_mock_model, device, "flow_matching").train()
        visual, map_input, vis, ego = _model_inputs(2, 8, device)
        out = model(visual, map_input, vis, ego, mode="train",
                    trajectory_target=torch.randn(2, TRAJ_DIM, device=device))
        assert torch.is_tensor(out) and out.shape == (2, TRAJ_DIM)

    def test_ignored_outside_train_mode(self, build_mock_model, device):
        model = self._model(build_mock_model, device, "flow_matching").eval()
        visual, map_input, vis, ego = _model_inputs(1, 8, device)
        out = model(visual, map_input, vis, ego, mode="infer",
                    return_planner_loss=True)
        assert torch.is_tensor(out) and out.shape == (1, TRAJ_DIM)

    def test_requires_trajectory_target(self, build_mock_model, device):
        model = self._model(build_mock_model, device, "flow_matching").train()
        visual, map_input, vis, ego = _model_inputs(1, 8, device)
        with pytest.raises(ValueError, match="trajectory_target"):
            model(visual, map_input, vis, ego, mode="train",
                  return_planner_loss=True)

    def test_bezier_rejects_flag(self, build_mock_model, device):
        model = self._model(build_mock_model, device, "bezier").train()
        visual, map_input, vis, ego = _model_inputs(1, 8, device)
        with pytest.raises(ValueError, match="compute_planner_loss"):
            model(visual, map_input, vis, ego, mode="train",
                  trajectory_target=torch.randn(1, TRAJ_DIM, device=device),
                  return_planner_loss=True)
