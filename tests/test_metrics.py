# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for src.metrics – the MAVIC-T evaluation metric module."""

import math

import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------

class TestComputeL1:
    def test_identical_images(self):
        from src.utils.metrics import compute_l1
        img = torch.rand(4, 3, 64, 64)
        assert compute_l1(img, img).item() == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        from src.utils.metrics import compute_l1
        a = torch.zeros(1, 1, 2, 2)
        b = torch.ones(1, 1, 2, 2)
        assert compute_l1(a, b).item() == pytest.approx(1.0, abs=1e-6)

    def test_output_is_scalar(self):
        from src.utils.metrics import compute_l1
        val = compute_l1(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
        assert val.dim() == 0


# ---------------------------------------------------------------------------
# Task score & overall score
# ---------------------------------------------------------------------------

class TestTaskScore:
    def test_perfect_score(self):
        from src.utils.metrics import task_score
        # All zeros → score should be 0
        assert task_score(0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_formula(self):
        from src.utils.metrics import task_score
        fid, lpips, l1 = 10.0, 0.3, 0.15
        expected = ((2.0 / math.pi) * math.atan(fid) + lpips + l1) / 3.0
        assert task_score(fid, lpips, l1) == pytest.approx(expected)

    def test_monotonicity_fid(self):
        from src.utils.metrics import task_score
        # Higher FID → higher (worse) score
        assert task_score(100, 0, 0) > task_score(1, 0, 0)


class TestOverallScore:
    def test_all_tasks_attempted(self):
        from src.utils.metrics import overall_score
        scores = {"sar2eo": 0.1, "sar2rgb": 0.2, "sar2ir": 0.3, "rgb2ir": 0.4}
        assert overall_score(scores) == pytest.approx(0.25)

    def test_missing_task_penalty(self):
        from src.utils.metrics import overall_score
        # 3 tasks attempted, 1 missing → penalty of 1
        scores = {"sar2eo": 0.0, "sar2rgb": 0.0, "sar2ir": 0.0}
        assert overall_score(scores) == pytest.approx(1.0)

    def test_no_tasks_attempted(self):
        from src.utils.metrics import overall_score
        assert overall_score({}) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# LPIPS wrapper
# ---------------------------------------------------------------------------

class TestLPIPS:
    def test_identical_images(self):
        from src.utils.metrics import LPIPS
        lpips = LPIPS(net_type="vgg")
        img = torch.rand(2, 3, 64, 64)
        val = lpips(img, img)
        assert val.item() == pytest.approx(0.0, abs=0.05)

    def test_grayscale_support(self):
        from src.utils.metrics import LPIPS
        lpips = LPIPS(net_type="vgg")
        img = torch.rand(2, 1, 64, 64)
        val = lpips(img, img)
        assert val.dim() == 0  # returns scalar


# ---------------------------------------------------------------------------
# MavicCriterion
# ---------------------------------------------------------------------------

class TestMavicCriterion:
    def test_identical_images_low_loss(self):
        from src.utils.metrics import MavicCriterion
        criterion = MavicCriterion(lpips_weight=1.0, l1_weight=1.0)
        img = torch.rand(2, 3, 64, 64)
        loss = criterion(img, img)
        assert loss.item() < 0.1

    def test_different_images_higher_loss(self):
        from src.utils.metrics import MavicCriterion
        criterion = MavicCriterion(lpips_weight=1.0, l1_weight=1.0)
        a = torch.zeros(2, 3, 64, 64)
        b = torch.ones(2, 3, 64, 64)
        loss = criterion(a, b)
        assert loss.item() > 0.5


# ---------------------------------------------------------------------------
# MetricResults
# ---------------------------------------------------------------------------

class TestMetricResults:
    def test_score_with_fid(self):
        from src.utils.metrics import MetricResults, task_score
        r = MetricResults(lpips=0.3, fid=10.0, l1=0.15)
        assert r.score == pytest.approx(task_score(10.0, 0.3, 0.15))

    def test_score_without_fid(self):
        from src.utils.metrics import MetricResults
        r = MetricResults(lpips=0.3, l1=0.15)
        assert r.score is None

    def test_to_dict(self):
        from src.utils.metrics import MetricResults
        r = MetricResults(lpips=0.3, fid=10.0, l1=0.15)
        d = r.to_dict()
        assert "lpips" in d and "fid" in d and "l1" in d and "score" in d

    def test_repr(self):
        from src.utils.metrics import MetricResults
        r = MetricResults(lpips=0.3, fid=10.0, l1=0.15)
        s = repr(r)
        assert "LPIPS" in s and "FID" in s and "L1" in s


# ---------------------------------------------------------------------------
# MetricCalculator
# ---------------------------------------------------------------------------

class TestMetricCalculator:
    def test_update_and_compute(self):
        from src.utils.metrics import MetricCalculator
        calc = MetricCalculator(device="cpu", compute_fid=False)
        img = torch.rand(2, 3, 64, 64)
        calc.update(img, img)
        result = calc.compute()
        assert result.l1 == pytest.approx(0.0, abs=1e-5)
        assert result.lpips < 0.1
        assert result.fid is None  # FID disabled

    def test_reset(self):
        from src.utils.metrics import MetricCalculator
        calc = MetricCalculator(device="cpu", compute_fid=False)
        img = torch.rand(2, 3, 64, 64)
        calc.update(img, img)
        calc.reset()
        result = calc.compute()
        # After reset, no data accumulated → defaults
        assert result.l1 == 0.0
        assert result.lpips == 0.0

    def test_multiple_batches(self):
        from src.utils.metrics import MetricCalculator
        calc = MetricCalculator(device="cpu", compute_fid=False)
        for _ in range(3):
            img = torch.rand(2, 3, 64, 64)
            calc.update(img, img)
        result = calc.compute()
        assert result.l1 == pytest.approx(0.0, abs=1e-5)

    def test_grayscale_input(self):
        from src.utils.metrics import MetricCalculator
        calc = MetricCalculator(device="cpu", compute_fid=False)
        img = torch.rand(2, 1, 64, 64)
        calc.update(img, img)
        result = calc.compute()
        assert result.l1 == pytest.approx(0.0, abs=1e-5)
