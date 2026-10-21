"""Tests for SoftCacheController and regrid_tail.

The reference functions below transcribe the authors' model-independent
``softcache/method.py`` (MIT, https://github.com/ShimonMalnick/softcache,
commit 49f00ab) and its Flux sampling loop, as an oracle.
"""

import mlx.core as mx
import pytest

from mlx_arsenal.diffusion import SoftCacheController, get_sampling_sigmas, regrid_tail


def _ref_decide(tau, jf_max, sigma_hi, raw_max, i, sigma, raw, acc):
    if i == 0 or acc >= tau:
        return "fresh", None
    if sigma <= sigma_hi and raw <= raw_max:
        headroom = max(0.0, 1.0 - acc / max(1e-9, tau))
        jf = 1.0 + (jf_max - 1.0) * headroom
        if jf > 1.01:
            return "jump", jf
    return "cache", None


def _ref_regrid(sigmas, i, target):
    n_tail = len(sigmas) - (i + 1)
    n_below = sum(1 for s in sigmas if 0.0 < s < target) + 1
    n_new = min(n_tail - 1, n_below)
    if n_new <= 1:
        return sigmas[: i + 1] + ([target, 0.0] if target > 0 else [0.0])
    tail = [target * (1.0 - k / (n_new - 1)) for k in range(n_new)]
    tail[0], tail[-1] = target, 0.0
    return sigmas[: i + 1] + tail


def _rel_l1(x, prev):
    return (mx.mean(mx.abs(x - prev)) / mx.mean(mx.abs(prev))).item()


def _toy(x, sigma):
    """Deterministic stand-ins for the block-stack input, residual and probe."""
    h = x * (1.0 + 0.5 * sigma)
    residual = mx.tanh(h) * (0.3 + sigma) - 0.1 * h * sigma**2
    probe = h * (1.0 - 0.8 * sigma) + 0.05
    return h, residual, probe


def _reference_loop(x, sigmas, tau, jf_max=8.0, beta=1.0):
    acc, prev_p = 0.0, None
    r_a = r_p = s_a = s_p = None
    actions, i = [], 0
    while sigmas[i] > 0:
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        h, residual, probe = _toy(x, sigma)
        raw = _rel_l1(probe, prev_p) if i > 0 else 0.0
        acc += raw
        prev_p = probe
        action, jf = _ref_decide(tau, jf_max, 0.9, 0.18, i, sigma, raw, acc)
        if action == "fresh":
            acc = 0.0
            if r_a is not None:
                r_p, s_p = r_a, s_a
            r_a, s_a = residual, sigma
            r = residual
        elif r_p is None:
            r = r_a
        else:
            assert s_a is not None and s_p is not None and r_a is not None
            lam = (sigma - s_a) / (s_a - s_p - 1e-8)
            r = r_a + (beta * lam) * (r_a - r_p)
        v = h + r
        if action == "jump":
            sigma_next = max(0.0, sigma + jf * (sigma_next - sigma))
            sigmas = _ref_regrid(sigmas, i, sigma_next)
        x = x + (sigma_next - sigma) * v
        actions.append(action[0].upper())
        i += 1
    return x, "".join(actions), sigmas


def _our_loop(x, sigmas, ctl):
    sigmas = list(sigmas)
    names = {"compute": "F", "reuse": "C", "jump": "J"}
    actions, i = [], 0
    while sigmas[i] > 0:
        h, residual, probe = _toy(x, sigmas[i])
        d = ctl.decide(i, sigmas[i], sigmas[i + 1], probe)
        if d.action == "compute":
            ctl.record(residual, sigmas[i])
            r = residual
        else:
            r = ctl.residual(sigmas[i])
        v = h + r
        if d.action == "jump":
            sigmas = regrid_tail(sigmas, i, d.sigma_next)
        x = x + (d.sigma_next - sigmas[i]) * v
        actions.append(names[d.action])
        i += 1
    return x, "".join(actions), sigmas


class TestSoftCacheLoop:
    @pytest.mark.parametrize("tau", [0.05, 0.3, 1.1])
    @pytest.mark.parametrize("shift", [1.0, 3.0])
    @pytest.mark.parametrize("beta", [1.0, 0.5])
    def test_matches_reference_loop(self, tau, shift, beta):
        sigmas = [float(s) for s in get_sampling_sigmas(50, shift=shift)]
        x0 = mx.random.normal((2, 16, 8), key=mx.random.key(0))
        ref_x, ref_actions, ref_sigmas = _reference_loop(x0, sigmas, tau, beta=beta)
        ctl = SoftCacheController(tau, beta=beta)
        x, actions, out_sigmas = _our_loop(x0, sigmas, ctl)
        assert actions == ref_actions
        # The reference divides the rel-L1 means in float32, RelL1State in float64;
        # a jump scales that rounding by up to 8x.
        assert out_sigmas == pytest.approx(ref_sigmas, abs=1e-5)
        assert mx.allclose(x, ref_x, atol=1e-4)

    def test_jumps_save_nodes(self):
        sigmas = [float(s) for s in get_sampling_sigmas(50, shift=3.0)]
        x0 = mx.random.normal((2, 16, 8), key=mx.random.key(0))
        _, actions, out_sigmas = _our_loop(x0, sigmas, SoftCacheController(0.3))
        assert "J" in actions
        assert len(actions) < 50
        assert out_sigmas[-1] == 0.0


class TestSoftCacheController:
    def test_first_node_computes(self):
        ctl = SoftCacheController(0.3)
        d = ctl.decide(0, 1.0, 0.98, mx.ones((4,)))
        assert d == ("compute", 0.98, 1.0)

    def test_budget_spent_computes_and_resets(self):
        ctl = SoftCacheController(0.3)
        ctl.decide(0, 1.0, 0.98, mx.ones((4,)))
        d = ctl.decide(1, 0.98, 0.96, mx.full((4,), 1.5))  # raw = 0.5 >= tau
        assert d.action == "compute"
        assert ctl.headroom == 1.0

    def test_budget_reached_exactly_computes(self):
        ctl = SoftCacheController(0.25)
        ctl.decide(0, 0.8, 0.78, mx.ones((4,)))
        assert ctl.decide(1, 0.78, 0.76, mx.full((4,), 1.25)).action == "compute"

    def test_stride_under_one_percent_is_not_a_jump(self):
        ctl = SoftCacheController(1.0, jump_probe_max=1.0)
        ctl.decide(0, 0.8, 0.78, mx.ones((4,)))
        d = ctl.decide(1, 0.78, 0.76, mx.full((4,), 1.999))  # headroom 0.001: stride 1.007
        assert d == ("reuse", 0.76, 1.0)

    def test_no_jump_at_high_sigma(self):
        ctl = SoftCacheController(0.3)
        ctl.decide(0, 1.0, 0.98, mx.ones((4,)))
        d = ctl.decide(1, 0.95, 0.93, mx.full((4,), 1.01))
        assert d == ("reuse", 0.93, 1.0)

    def test_no_jump_on_sharp_probe_change(self):
        ctl = SoftCacheController(1.0)
        ctl.decide(0, 0.8, 0.78, mx.ones((4,)))
        d = ctl.decide(1, 0.78, 0.76, mx.full((4,), 1.2))  # raw = 0.2 > 0.18
        assert d.action == "reuse"

    def test_jump_stride_follows_headroom(self):
        ctl = SoftCacheController(1.0)
        ctl.decide(0, 0.8, 0.78, mx.ones((4,)))
        d = ctl.decide(1, 0.6, 0.58, mx.full((4,), 1.1))  # raw = 0.1, headroom 0.9
        assert d.action == "jump"
        assert d.jump_factor == pytest.approx(1 + 7 * 0.9)
        assert d.sigma_next == pytest.approx(0.6 - 0.02 * (1 + 7 * 0.9))
        assert ctl.headroom == pytest.approx(0.9)

    def test_jump_target_clamped_at_zero(self):
        ctl = SoftCacheController(1.0)
        ctl.decide(0, 0.1, 0.05, mx.ones((4,)))
        d = ctl.decide(1, 0.05, 0.02, mx.full((4,), 1.01))
        assert d.action == "jump" and d.sigma_next == 0.0

    def test_residual_motion(self):
        ctl = SoftCacheController(0.3, beta=0.5)
        ctl.record(mx.array([1.0]), 0.8)
        assert ctl.residual(0.7).item() == 1.0  # one anchor: reused as is
        ctl.record(mx.array([2.0]), 0.6)
        lam = (0.5 - 0.6) / (0.6 - 0.8 - 1e-8)
        assert ctl.residual(0.5).item() == pytest.approx(2.0 + 0.5 * lam * 1.0)

    def test_residual_before_record_raises(self):
        with pytest.raises(RuntimeError, match="record"):
            SoftCacheController(0.3).residual(0.5)

    def test_decide_before_first_node_raises(self):
        with pytest.raises(RuntimeError, match="step 0"):
            SoftCacheController(0.3).decide(1, 0.9, 0.8, mx.ones((4,)))

    def test_degenerate_probe_computes(self):
        ctl = SoftCacheController(0.3)
        ctl.decide(0, 1.0, 0.98, mx.zeros((4,)))
        assert ctl.decide(1, 0.98, 0.96, mx.ones((4,))).action == "compute"

    def test_reset(self):
        ctl = SoftCacheController(0.3)
        ctl.decide(0, 1.0, 0.98, mx.ones((4,)))
        ctl.record(mx.array([1.0]), 1.0)
        ctl.reset()
        with pytest.raises(RuntimeError):
            ctl.residual(0.5)

    @pytest.mark.parametrize("kw", [{"tau": 0.0}, {"tau": 0.3, "max_jump_factor": 0.5}])
    def test_rejects_bad_config(self, kw):
        with pytest.raises(ValueError):
            SoftCacheController(**kw)


class TestRegridTail:
    @pytest.mark.parametrize("index", [3, 10, 30])
    @pytest.mark.parametrize("target_frac", [0.0, 0.2, 0.5, 0.95])
    def test_matches_reference(self, index, target_frac):
        sigmas = [float(s) for s in get_sampling_sigmas(40, shift=3.0)]
        target = sigmas[index + 1] * target_frac
        assert regrid_tail(sigmas, index, target) == _ref_regrid(sigmas, index, target)

    def test_shortens_and_ends_at_zero(self):
        sigmas = [1.0 - k / 20 for k in range(21)]
        out = regrid_tail(sigmas, 5, 0.5)
        assert out[:6] == sigmas[:6]
        assert out[6] == 0.5 and out[-1] == 0.0
        assert len(out) < len(sigmas)
        assert all(a > b for a, b in zip(out, out[1:], strict=False))
