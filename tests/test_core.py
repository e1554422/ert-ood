"""核心数值单测: 论文初稿中的示例 + NIG/Student-t 对齐 + 残差化 + 划分无泄漏。"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats
from scipy.special import digamma, gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ert.data import GUARD, fit_calib_split, kfold_ids, per_class_subset  # noqa: E402
from ert.heads import edl_loss, kl_dirichlet_uniform  # noqa: E402
from ert.scores import ResidualFusion, effective_layers  # noqa: E402
from ert.state import anchor_residuals, class_routes, evidence_state, summaries  # noqa: E402
from ert.transition import nig_epistemic, nig_nll  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_paper_vacuity_examples():
    # 初稿: α=[1,1] -> u=1 (纯空缺); α=[100,100] -> u=0.01, p=[.5,.5] 熵最大 (纯冲突)
    la = np.log(np.array([[[1.0, 1.0], [100.0, 100.0]]], dtype=np.float32))
    s = summaries(la, device=DEV)
    assert s["u"][0, 0] == pytest.approx(1.0)
    assert s["u"][0, 1] == pytest.approx(0.01)
    assert s["H"][0, 1] == pytest.approx(math.log(2), rel=1e-5)


def test_kl_dirichlet_uniform_matches_closed_form():
    a = np.array([2.0, 0.5, 3.0])
    s = a.sum()
    ref = gammaln(s) - gammaln(3) - gammaln(a).sum() + ((a - 1) * (digamma(a) - digamma(s))).sum()
    got = kl_dirichlet_uniform(torch.tensor(a)).item()
    assert got == pytest.approx(ref, rel=1e-6)
    assert kl_dirichlet_uniform(torch.ones(5)).item() == pytest.approx(0.0, abs=1e-6)


def test_edl_losses_prefer_correct_evidence():
    y = torch.tensor([0])
    good = torch.tensor([[50.0, 1.0, 1.0]])
    bad = torch.tensor([[1.0, 50.0, 1.0]])
    for kind in ("mse", "digamma", "log"):
        assert edl_loss(good, y, kind, 1.0) < edl_loss(bad, y, kind, 1.0)


def test_nig_nll_equals_student_t():
    rng = np.random.default_rng(0)
    g, n, a, b, y = (rng.uniform(0.5, 3, 50) for _ in range(5))
    ours = nig_nll(*(torch.tensor(v, dtype=torch.float64) for v in (y, g, n, a, b))).numpy()
    scale = np.sqrt(b * (1 + n) / (n * a))
    ref = -stats.t.logpdf(y, df=2 * a, loc=g, scale=scale)
    np.testing.assert_allclose(ours, ref, rtol=1e-8)
    assert torch.all(nig_epistemic(torch.tensor(n), torch.tensor(a + 1.0), torch.tensor(b)) > 0)


def test_routes_and_hard_anchor_residual_zero_at_class_mean():
    # 每类只有一个样本时, 硬锚定到真实类 (末层正确) 的残差应恒为 0
    C = 4
    la = np.log(np.eye(C, dtype=np.float32) * 20 + 1)[None].repeat(3, 0)  # [L=3, N=4, C]
    y = np.arange(C)
    nu = class_routes(la, y, C, device=DEV)
    r, anchors = anchor_residuals(la, nu, "hard", device=DEV, out_dtype=np.float32)
    np.testing.assert_array_equal(anchors, y)
    np.testing.assert_allclose(r, 0, atol=1e-5)
    v = evidence_state(torch.from_numpy(la[0]))
    assert v.shape == (C, C + 1)


def test_residual_fusion_removes_linear_redundancy():
    rng = np.random.default_rng(0)
    U = rng.normal(size=5000)
    G = 2 * U + rng.normal(size=5000) * 0.1
    D = rng.normal(size=5000)
    fus = ResidualFusion(["U", "G", "D"], {"G": 0.3, "D": 0.3}).fit({"U": U, "G": G, "D": D})
    parts = fus.components({"U": U, "G": G, "D": D})
    assert abs(np.corrcoef(parts["U"], parts["G"])[0, 1]) < 1e-6
    assert abs(np.corrcoef(parts["G"], parts["D"])[0, 1]) < 1e-6


def test_effective_layers_is_tail_contiguous():
    acc = np.array([.001, .002, .01, .2, .03, .3, .5, .6, .7, .75, .8, .82])
    assert effective_layers(acc, 0.05) == [5, 6, 7, 8, 9, 10, 11]


def test_splits_disjoint_and_stratified():
    y = np.repeat(np.arange(50), 100)
    fit, cal = fit_calib_split(y, seed=1)
    assert not set(fit) & set(cal) and len(fit) + len(cal) == len(y)
    assert np.all(np.bincount(y[cal]) == 20)
    folds = kfold_ids(y[fit], 5, seed=1)
    assert set(np.unique(folds)) == set(range(5))
    sub = per_class_subset(y, 50, seed=0)
    assert np.all(np.bincount(y[sub]) == 50)


def test_ood_guard_blocks_selection_access():
    with pytest.raises(RuntimeError):
        with GUARD:
            GUARD.touch("ninco")
