from __future__ import annotations

import math
from dataclasses import dataclass

_CONTINUOUS_KEYS = ("root_total_cost_log", "root_rows_log", "root_width_log", "node_count_log",
                    "max_depth", "sum_cost_log", "sum_rows_log", "max_rows_log", "mean_width_log")


@dataclass(frozen=True)
class DomainAssessment:
    rms_z_distance: float
    max_z_distance: float
    outside_training_range_count: int
    unseen_structural_features: tuple[str, ...]


@dataclass(frozen=True)
class FeatureDomain:
    means: dict[str, float]
    scales: dict[str, float]
    minimums: dict[str, float]
    maximums: dict[str, float]
    structural_features: frozenset[str]

    @classmethod
    def fit(cls, feature_rows: list[dict[str, float]]) -> "FeatureDomain":
        if not feature_rows:
            raise ValueError("at least one feature row is required")
        means, scales, minimums, maximums = {}, {}, {}, {}
        for key in _CONTINUOUS_KEYS:
            values = [float(row.get(key, 0.0)) for row in feature_rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            means[key] = mean; scales[key] = max(math.sqrt(variance), 0.25)
            minimums[key] = min(values); maximums[key] = max(values)
        structural = frozenset(key for row in feature_rows for key in row
                               if key.startswith("op=") or key.startswith("join="))
        return cls(means, scales, minimums, maximums, structural)

    def assess(self, features: dict[str, float]) -> DomainAssessment:
        z = [abs((float(features.get(k, 0.0)) - self.means[k]) / self.scales[k]) for k in _CONTINUOUS_KEYS]
        unseen = tuple(sorted(k for k in features if (k.startswith("op=") or k.startswith("join="))
                              and k not in self.structural_features and float(features[k]) != 0.0))
        outside = sum(float(features.get(k, 0.0)) < self.minimums[k] or
                      float(features.get(k, 0.0)) > self.maximums[k] for k in _CONTINUOUS_KEYS)
        return DomainAssessment(math.sqrt(sum(v * v for v in z) / len(z)), max(z, default=0.0), outside, unseen)
