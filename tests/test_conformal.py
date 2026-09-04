import pytest

from aster.uncertainty import ConformalLogCalibrator


def test_conformal_interval_uses_finite_sample_quantile_and_covers_calibration_points():
    predicted=[10.0,20.0,30.0,40.0]
    stds=[0.1,0.1,0.1,0.1]
    actual=[11.0,18.0,33.0,44.0]
    calibrator=ConformalLogCalibrator.fit(predicted,stds,actual,alpha=0.25,min_log_scale=0.05)
    assert calibrator.calibration_examples == 4
    assert calibrator.target_coverage == pytest.approx(0.75)
    assert calibrator.quantile > 0
    covered=sum(calibrator.covers(p,s,a) for p,s,a in zip(predicted,stds,actual,strict=True))
    assert covered >= 3


def test_conformal_scale_floor_prevents_zero_width_overconfidence():
    calibrator=ConformalLogCalibrator.fit([10.0,20.0],[0.0,0.0],[12.0,18.0],alpha=0.2,min_log_scale=0.05)
    interval=calibrator.interval(15.0,0.0)
    assert interval.lower_ms < 15.0 < interval.upper_ms
    assert interval.log_radius > 0


def test_conformal_rejects_invalid_inputs():
    with pytest.raises(ValueError,match="equal length"):
        ConformalLogCalibrator.fit([1.0],[0.1],[1.0,2.0])
    with pytest.raises(ValueError,match="at least two"):
        ConformalLogCalibrator.fit([1.0],[0.1],[1.0])
    with pytest.raises(ValueError,match="positive"):
        ConformalLogCalibrator.fit([0.0,1.0],[0.1,0.1],[1.0,1.0])
