import argparse
import logging
import sys
import matplotlib
sys.path.append("/lustre/gh/main/clv9")
from hot_baseline_worker import run_diagnostics

# Run in py38_orca environment

args = argparse.Namespace(
    ms        = "/fast/rbyrne/20260407_123010-123201_52MHz_calibrated.ms",       # Path to your MS file
    col       = "CORRECTED_DATA",     # or "DATA" if uncalibrated
    sigma     = 5.0,                  # Sigma threshold for hot baseline detection
    threshold = 0.10,                 # Fraction of baselines to trigger antenna flag (10%)
    uv_cut    = 0.0,                  # Short baseline cut in meters (0 = off)
    uv_cut_lambda = 0.0,              # Short baseline cut in wavelengths (0 = off)
    run_uv    = True,                 # Run amplitude-vs-UV analysis
    uv_sigma  = 5.0,                  # Sigma for UV outlier detection
    uv_window_size = 100,             # Rolling window size for UV analysis

    apply_antenna_flags  = True,      # Write antenna flags back to MS
    apply_baseline_flags = True,      # Write baseline flags back to MS
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

run_diagnostics(args, logger)