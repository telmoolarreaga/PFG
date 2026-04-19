# amiaire/config.py
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DEFAULT_INPUT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_REGRESSION_PARAMS_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "regression_params.json")
DEFAULT_CALIBRATION_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# --- Image Processing Parameters ---
# Note: all pixel-based parameters scaled ×2 to match the new 2000×2000 resolution
# (previously 1000×1000). Area-based parameters scaled ×4 (area grows as resolution²).
ROI_EXTRACTION_SETTINGS = {
    "gaussian_blur_kernel":       (17, 17),  # previously (9, 9), scaled ×2 (must be odd)
    "adaptive_thresh_block_size": 241,       # previously 121, scaled ×2 (must be odd)
    "adaptive_thresh_c":          2,         # dimensionless, unchanged
    "morph_open_kernel":          (21, 21),  # previously (11, 11), scaled ×2
    "morph_close_kernel":         (41, 41),  # previously (21, 21), scaled ×2
    "contour_area_min_ratio":     0.1,       # ratio of image area, unchanged
    "contour_area_max_ratio":     0.7,       # ratio of image area, unchanged
    "contour_similarity_threshold": 0.85,   # dimensionless, unchanged
    "corner_refine_window_size":  (5, 5),    # subpixel refinement, unchanged
    "corner_adjust_margin":       30         # previously 15, scaled ×2
}

GRAYSCALE_CONVERSION_PARAMS = {
    "target_size": (2000, 2000)              # previously (1000, 1000)
}

BACKGROUND_IMPROVEMENT_PARAMS = {
    "kernel_size": (41, 41),                 # previously (21, 21), scaled ×2
    "sigma":       20.0                      # previously 10.0, scaled ×2
}

CLAHE_PARAMS = {
    "clip_limit": 0.004,                     # dimensionless, unchanged
    "nbins":      12                         # dimensionless, unchanged
}

RESCALE_INTENSITY_PARAMS = {
    "in_range_percent": (0, 20)              # percentile-based, unchanged
}

SAUVOLA_THRESHOLD_PARAMS = {
    "window_size": 41,                       # previously 21, scaled ×2 (must be odd)
    "k":           0.18,                     # dimensionless, unchanged
    "invert":      True
}

# --- Particle Analysis Parameters ---
DEFAULT_FILTER_PARAMETERS = {
    'min_area':         12.0,  # previously 3.0,  scaled ×4 (3px² @ 1000→12px² @ 2000)
    'max_area':         1200,  # previously 300,   scaled ×4
    'min_solidity':     0.3,   # dimensionless, unchanged
    'max_solidity':     1.0,   # dimensionless, unchanged
    'min_aspect_ratio': 0.00,  # dimensionless, unchanged
    'max_aspect_ratio': 4.0,   # dimensionless, unchanged
    'min_feret':        0.00,  # unchanged
    'max_feret':        100.0  # previously 50.0, scaled ×2
}

# --- Pollution Level Parameters ---
POLLUTION_CALCULATION_PARAMS = {
    "papersensor_size": (0.06, 0.06),        # meters, physical constant — unchanged
    "PM25": {
        "particle_diameter_microns": 2.5,
        "particle_density_g_cm3":    1.65
    },
    "PM10": {
        "particle_diameter_microns": 10.0,
        "particle_density_g_cm3":    1.65
    }
}