"""Bearing geometry and characteristic fault frequencies (LF-PNCN).

f_o = z*f_r/2 * (1 - (d/D) cos th)             outer-race (BPFO)
f_i = z*f_r/2 * (1 + (d/D) cos th)             inner-race (BPFI)
f_b = D*f_r/(2d) * (1 - ((d/D) cos th)^2)      ball spin (BSF)
f_c = f_r/2 * (1 - (d/D) cos th)               cage (FTF)
"""
import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Bearing:
    name: str
    z: int            # number of rolling elements
    d: float          # rolling element diameter [mm]
    D: float          # pitch circle diameter [mm]
    theta_deg: float = 0.0  # contact angle [deg]


# Lessmeier et al. 2016, Table 1 (6203, FAG): pitch circle 28.55 mm, 8 elements,
# element diameter 6.75 mm, nominal pressure angle 0. See README for the 29.05 mm discrepancy.
PU_6203 = Bearing("PU_6203", 8, 6.75, 28.55, 0.0)
# Alternative value (D = 29.05 mm); kept for sensitivity checks only.
PU_6203_SPEC = Bearing("PU_6203_D29.05", 8, 6.75, 29.05, 0.0)
# CWRU drive-end 6205-2RS JEM SKF.
CWRU_6205 = Bearing("CWRU_6205", 9, 7.94, 39.04, 0.0)


def fault_frequencies(f_r, bearing):
    """Return dict with f_o, f_i, f_b, f_c [Hz] for shaft frequency f_r [Hz]."""
    r = (bearing.d / bearing.D) * math.cos(math.radians(bearing.theta_deg))
    z = bearing.z
    return {
        "f_o": z * f_r / 2.0 * (1.0 - r),
        "f_i": z * f_r / 2.0 * (1.0 + r),
        "f_b": bearing.D * f_r / (2.0 * bearing.d) * (1.0 - r * r),
        "f_c": f_r / 2.0 * (1.0 - r),
    }


def shaft_frequency_from_name(stem):
    """Nominal shaft frequency [Hz] from a PU file name such as N15_M07_F10_K001_1 (N15=1500 rpm)."""
    m = re.match(r"^N(\d\d)_", stem)
    if not m:
        raise ValueError(f"cannot read speed from {stem}")
    return int(m.group(1)) * 100.0 / 60.0


def shaft_frequency_from_rpm(rpm):
    return float(rpm) / 60.0
