"""Exact persistent-homology correctness."""
import numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from topofuse.topo.ph import betti_numbers
from topofuse.topo.matching import bottleneck
from topofuse.losses.topo_projection import TopologyProjection
from topofuse.topo.pseudo_diagram import oracle_diagram_from_label


def test_exact_betti():
    m = np.zeros((24, 24, 24)); m[4:9, 4:9, 4:9] = 1; m[14:19, 14:19, 14:19] = 1
    assert betti_numbers(m)[0] == 2          # two components


def test_bottleneck_orientation():
    # superlevel points are below-diagonal; must not collapse to 0
    a = np.array([[0.99, 0.02], [0.99, 0.02]])
    b = np.array([[1.0, 0.0]])
    assert bottleneck(a, b) > 0.3


def test_projection_edits():
    Z = torch.full((1, 2, 32, 32, 32), -6.0); Z[:, 0] = 6.0
    for (z, y, x) in [(5, 5, 5), (5, 5, 25), (5, 25, 5), (25, 5, 5), (25, 25, 25), (15, 15, 15)]:
        Z[0, 1, z-2:z+3, y-2:y+3, x-2:x+3] = 8.0
    lab = np.zeros((32, 32, 32), np.int64); lab[12:20, 12:20, 12:20] = 1
    tgt = [[oracle_diagram_from_label(lab, c, 2, 0.05) for c in range(2)]]
    proj = TopologyProjection(num_classes=2, T_max=5, epsilon=0.05, downsample_s=2)
    Zp, certs = proj(Z.clone(), target_diagrams=tgt)
    assert certs[0].spatial_sparsity > 0     # real critical-voxel edits happened
    assert (Zp != Z).any()
