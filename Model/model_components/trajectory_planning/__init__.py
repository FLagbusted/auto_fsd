from .base import BasePlanner
from .flow_matching_planner import FlowMatchingPlanner
from .bezier_planner import BezierPlanner
from .gru_planner import GRUPlanner

PLANNER_REGISTRY = {
    "flow_matching": FlowMatchingPlanner,
    "bezier": BezierPlanner,
    "gru": GRUPlanner,
}

def build_planner(planner_mode, **kwargs):
    """Construct a planner by name.

    Forwards ``**kwargs`` directly to the planner __init__; planners ignore
    args they do not understand by accepting only what they need (the
    registry does not pre-filter, so callers must pass kwargs the chosen
    planner accepts).
    """
    if planner_mode not in PLANNER_REGISTRY:
        raise ValueError(
            f"Unknown planner_mode {planner_mode!r}. "
            f"Available: {sorted(PLANNER_REGISTRY)}."
        )
    # flow_matching's forward() Euler-integrates from fresh noise, so it must be
    # trained with FlowMatchingPlanner.compute_planner_loss (velocity MSE), not
    # by regressing forward()'s output — that drives it to the conditional mean.
    # train_il(planner_mode="flow_matching") does this via
    # AutoE2E.forward(..., return_planner_loss=True); custom loops must too.
    return PLANNER_REGISTRY[planner_mode](**kwargs)

__all__ = [
    "BasePlanner",
    "FlowMatchingPlanner",
    "GRUPlanner",
    "BezierPlanner",
    "PLANNER_REGISTRY",
    "build_planner",
]
