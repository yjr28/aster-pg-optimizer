from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeInterval:
    lower_ms: float
    upper_ms: float
    log_radius: float


@dataclass(frozen=True)
class ConformalLogCalibrator:
    """Finite-sample split-conformal calibration for positive runtimes.

    Aster predicts log1p(runtime). The nonconformity score is the absolute log-space
    residual divided by ensemble disagreement, with a floor so overconfident ensembles
    cannot produce zero-width intervals. The finite-sample conformal quantile is then
    used as a multiplicative radius in that standardized log space.
    """

    alpha: float
    quantile: float
    calibration_examples: int
    min_log_scale: float = 0.05

    @classmethod
    def fit(
        cls,
        predicted_runtime_ms: list[float] | tuple[float, ...],
        log_stds: list[float] | tuple[float, ...],
        actual_runtime_ms: list[float] | tuple[float, ...],
        *,
        alpha: float = 0.10,
        min_log_scale: float = 0.05,
    ) -> "ConformalLogCalibrator":
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if min_log_scale <= 0:
            raise ValueError("min_log_scale must be positive")
        if not (
            len(predicted_runtime_ms) == len(log_stds) == len(actual_runtime_ms)
        ):
            raise ValueError("prediction, uncertainty, and label arrays must have equal length")
        if len(actual_runtime_ms) < 2:
            raise ValueError("at least two calibration examples are required")

        scores: list[float] = []
        for predicted, log_std, actual in zip(
            predicted_runtime_ms, log_stds, actual_runtime_ms, strict=True
        ):
            if predicted <= 0 or actual <= 0:
                raise ValueError("predicted and actual runtimes must be positive")
            if log_std < 0 or not math.isfinite(log_std):
                raise ValueError("log_std values must be finite and non-negative")
            residual = abs(math.log1p(actual) - math.log1p(predicted))
            scale = max(float(log_std), min_log_scale)
            scores.append(residual / scale)

        scores.sort()
        n = len(scores)
        rank = math.ceil((n + 1) * (1 - alpha))
        index = min(n, max(1, rank)) - 1
        return cls(
            alpha=alpha,
            quantile=float(scores[index]),
            calibration_examples=n,
            min_log_scale=min_log_scale,
        )

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    def interval(self, runtime_ms: float, log_std: float) -> RuntimeInterval:
        if runtime_ms <= 0:
            raise ValueError("runtime_ms must be positive")
        if log_std < 0 or not math.isfinite(log_std):
            raise ValueError("log_std must be finite and non-negative")
        radius = self.quantile * max(float(log_std), self.min_log_scale)
        center = math.log1p(runtime_ms)
        lower = max(1e-9, math.expm1(center - radius))
        upper = max(lower, math.expm1(center + radius))
        return RuntimeInterval(lower_ms=lower, upper_ms=upper, log_radius=radius)

    def covers(self, runtime_ms: float, log_std: float, actual_runtime_ms: float) -> bool:
        interval = self.interval(runtime_ms, log_std)
        return interval.lower_ms <= actual_runtime_ms <= interval.upper_ms
