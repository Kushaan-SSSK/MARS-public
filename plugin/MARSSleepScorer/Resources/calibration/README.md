# Runtime Calibration

MARS requires an AccuSleePy calibration CSV at runtime. Do not enable hardware
TTL output until the calibration file has been selected and validated for the
current animal and rig.

The bridge validates that the file contains the AccuSleePy mixture mean and
standard deviation columns. The native processor blocks acquisition when
hardware output is enabled and the calibration file is missing.
