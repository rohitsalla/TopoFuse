"""End-to-end smoke: forward -> 4-term loss -> backward, plus inference path."""
import torch, warnings
warnings.filterwarnings("ignore")
from topofuse.models.topofuse import TopoFuse


def _batch(C=2, D=24):
    x = torch.randn(1, 1, D, D, D)
    y = torch.zeros(1, D, D, D, dtype=torch.long)
    y[:, 4:12, 4:12, 4:12] = 1
    gt = {"beta0": torch.tensor([[0.0, 1.0]]), "beta2": torch.tensor([[0.0, 0.0]]),
          "budgets": torch.zeros(1, C, 2, 6)}
    return x, y, gt


def test_train_step():
    m = TopoFuse(num_classes=2, feature_dim=16, slice_size=32, T_max=2, sam_checkpoint=None)
    m.train()
    x, y, gt = _batch()
    out = m(x, labels=y)
    losses = m.compute_loss(out, y, gt_topology=gt)
    assert set(losses) >= {"total", "main", "aux", "topo", "prior"}
    losses["total"].backward()
    g = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    assert g > 0
    assert out["logits_post"].shape == (1, 2, 24, 24, 24)
    assert len(out["certificates"]) == 1


def test_inference_path():
    m = TopoFuse(num_classes=2, feature_dim=16, slice_size=32, T_max=2, sam_checkpoint=None)
    m.eval()
    x, y, _ = _batch()
    with torch.no_grad():
        out = m(x)                       # no labels -> prior-driven projection
    assert out["prob_post"].shape == (1, 2, 24, 24, 24)
