"""TopoFuse persistent-homology backend (exact, CubicalRipser + GUDHI)."""
from .ph import (
    Diagram, Feature, DIMS,
    compute_ph_raw, differentiable_diagram, betti_numbers,
)
from .matching import (
    bottleneck, optimal_matching, unmatched_features, wasserstein_cost,
)
from .pseudo_diagram import (
    TargetDiagram, BUDGET_THRESHOLDS,
    build_pseudo_diagram, oracle_diagram_from_label,
)
from .descriptor import TopologyDescriptor

__all__ = [
    "Diagram", "Feature", "DIMS",
    "compute_ph_raw", "differentiable_diagram", "betti_numbers",
    "bottleneck", "optimal_matching", "unmatched_features", "wasserstein_cost",
    "TargetDiagram", "BUDGET_THRESHOLDS",
    "build_pseudo_diagram", "oracle_diagram_from_label",
    "TopologyDescriptor",
]
