# main/analysis.py
import cv2
import numpy as np
from skimage import measure
import os
import json
from . import config
import pandas as pd
from skimage.filters import sobel
from skimage.measure import perimeter

# Evitar warnings numéricos
np.seterr(divide='ignore', invalid='ignore')


def extract_features_per_particle(filtered_regions, original_img=None, min_area=0):

    import numpy as np
    import cv2
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from scipy.stats import entropy

    features_list = []

    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if original_img is not None else None

    for region in filtered_regions:

        if region.area < min_area:
            continue

        coords = region.coords

        intensities = gray[coords[:,0], coords[:,1]] if gray is not None else np.array([0])

        area = region.area
        perim = region.perimeter
        major = region.axis_major_length
        minor = region.axis_minor_length
        solidity = region.solidity
        extent = region.extent
        convex_area = region.area_convex
        ecc = region.eccentricity

        aspect_ratio = major/minor if minor>0 else 1

        equivalent_diameter = region.equivalent_diameter_area

        feret_max = getattr(region,'feret_diameter_max', equivalent_diameter)
        feret_min = minor

        compactness = (perim**2)/area if area>0 else 0
        roundness = 4*np.pi*area/(perim**2) if perim>0 else 0

        sphericity = roundness

        rectangularity = area/(major*minor) if major*minor>0 else 0

        convex_ratio = area/convex_area if convex_area>0 else 0

        area_perimeter_ratio = area/perim if perim>0 else 0

        # intensidad
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)

        intensity_range = np.max(intensities)-np.min(intensities)

        hist,_ = np.histogram(intensities,bins=32)

        intensity_entropy = entropy(hist+1)

        contrast_internal = std_intensity/mean_intensity if mean_intensity>0 else 0

        local_variance = np.var(intensities)

        # GLCM
        if gray is not None:

            minr,minc,maxr,maxc = region.bbox
            patch = gray[minr:maxr,minc:maxc]

            glcm = graycomatrix(patch,[1],[0],levels=256,symmetric=True,normed=True)

            glcm_contrast = graycoprops(glcm,'contrast')[0,0]
            glcm_homogeneity = graycoprops(glcm,'homogeneity')[0,0]
            glcm_energy = graycoprops(glcm,'energy')[0,0]
            glcm_correlation = graycoprops(glcm,'correlation')[0,0]

        else:

            glcm_contrast=glcm_homogeneity=glcm_energy=glcm_correlation=0

        # LBP
        if gray is not None:

            lbp = local_binary_pattern(patch,8,1)

            lbp_mean = np.mean(lbp)
            lbp_std = np.std(lbp)

        else:

            lbp_mean=lbp_std=0

        # HU moments
        mask = region.image.astype(np.uint8)

        moments = cv2.moments(mask)

        hu = cv2.HuMoments(moments).flatten()

        # topologia
        euler_number = region.euler_number
        holes = 1 - euler_number

        # contorno

        mask = region.image.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = contours[0]
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
        else:
            hull_area = area
        convexity_defects = max(0, hull_area - area)
        hull_ratio = area/hull_area if hull_area>0 else 0
        roughness_index = perim/(2*np.sqrt(np.pi*area)) if area>0 else 0

    

        # fractal dimension simple
        if area > 1 and perim > 0:
            fractal_dimension = np.log(perim)/np.log(area)
        else:
            fractal_dimension = np.nan

        features_list.append({

        'area':area,
        'perimeter':perim,

        'major_axis':major,
        'minor_axis':minor,
        'aspect_ratio':aspect_ratio,
        'eccentricity':ecc,

        'equivalent_diameter':equivalent_diameter,
        'feret_max':feret_max,
        'feret_min':feret_min,

        'solidity':solidity,
        'extent':extent,
        'convex_area':convex_area,

        'compactness':compactness,
        'roundness':roundness,
        'sphericity':sphericity,
        'rectangularity':rectangularity,

        'convex_ratio':convex_ratio,
        'area_perimeter_ratio':area_perimeter_ratio,

        'mean_intensity':mean_intensity,
        'std_intensity':std_intensity,
        'intensity_range':intensity_range,
        'contrast_internal':contrast_internal,
        'intensity_entropy':intensity_entropy,
        'local_variance':local_variance,

        'glcm_contrast':glcm_contrast,
        'glcm_homogeneity':glcm_homogeneity,
        'glcm_energy':glcm_energy,
        'glcm_correlation':glcm_correlation,

        'lbp_mean':lbp_mean,
        'lbp_std':lbp_std,

        'hu1':hu[0],
        'hu2':hu[1],
        'hu3':hu[2],
        'hu4':hu[3],
        'hu5':hu[4],
        'hu6':hu[5],
        'hu7':hu[6],

        'fractal_dimension':fractal_dimension,
        'convexity_defects':convexity_defects,
        'hull_ratio':hull_ratio,
        'roughness_index':roughness_index,

        'euler_number':euler_number,
        'holes':holes

        })

    
    return pd.DataFrame(features_list)

def analyze_particles(binary_image_path: str,
                      original_image_path: str | None = None,
                      output_dir: str | None = None,
                      filter_params: dict | None = None):
    """
    Analiza partículas en una imagen binaria, filtra según parámetros, genera overlays,
    y extrae features. Devuelve diccionario con estadísticas y lista de regiones filtradas.
    """
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    if binary_image is None:
        raise FileNotFoundError(f"Binary image not found: {binary_image_path}")

    if filter_params is None:
        filter_params = config.DEFAULT_FILTER_PARAMETERS

    original_img = cv2.imread(original_image_path) if original_image_path else None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ---------------- LABELING ----------------
    binary_bool = binary_image > 0
    label_image = measure.label(binary_bool)
    regions = measure.regionprops(label_image)

    # ---------------- DEBUG OVERLAY ----------------
    if original_img is not None and output_dir:
        overlay_all = original_img.copy()
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            cv2.rectangle(overlay_all, (minc, minr), (maxc, maxr), (255, 0, 0), 1)
        cv2.imwrite(os.path.join(output_dir, 'debug_all_particles_overlay.png'), overlay_all)

    # ---------------- FILTRADO ----------------
    filtered_regions = []
    for region in regions:
        area = region.area
        solidity = region.solidity
        major = region.axis_major_length
        minor = region.axis_minor_length
        aspect_ratio = major / minor if minor > 0 else 1
        feret_diameter = getattr(region, 'feret_diameter_max', region.equivalent_diameter_area)

        if not (filter_params.get('min_area', 0) <= area <= filter_params.get('max_area', float('inf'))): continue
        if not (filter_params.get('min_solidity', 0) <= solidity <= filter_params.get('max_solidity', 1.0)): continue
        if not (filter_params.get('min_aspect_ratio', 0) <= aspect_ratio <= filter_params.get('max_aspect_ratio', float('inf'))): continue
        if not (filter_params.get('min_feret', 0) <= feret_diameter <= filter_params.get('max_feret', float('inf'))): continue

        filtered_regions.append(region)

    # ---------------- MÁSCARA FILTRADA ----------------
    filtered_mask = np.zeros(label_image.shape, dtype=np.uint8)
    for region in filtered_regions:
        coords = region.coords
        filtered_mask[coords[:, 0], coords[:, 1]] = 255

    if output_dir:
        cv2.imwrite(os.path.join(output_dir, 'filtered_binary_mask.png'), filtered_mask)

    # ---------------- FEATURES ----------------
    df_particles = extract_features_per_particle(filtered_regions, original_img, min_area=filter_params.get('min_area', 0))

    if output_dir:
        df_path = os.path.join(output_dir, "particle_features.csv")
        df_particles.to_csv(df_path, index=False)
        print(f"Particle features saved to {df_path}")

    return {
        'num_particles': len(filtered_regions),
        'total_particle_area_pixels': np.count_nonzero(filtered_mask),
        'image_area_pixels': filtered_mask.shape[0] * filtered_mask.shape[1],
        'area_percentage': (np.count_nonzero(filtered_mask) / (filtered_mask.shape[0]*filtered_mask.shape[1])) * 100
                          if filtered_mask.shape[0]*filtered_mask.shape[1] > 0 else 0,
        'filtered_regions': filtered_regions,
        'original_img': original_img
    }


def calculate_pollution_level(analysis_results: dict,
                              model_type: str,
                              regression_models_path: str = config.DEFAULT_REGRESSION_PARAMS_PATH):
    """
    Calcula concentración de contaminación usando el análisis de partículas y modelo de regresión.
    """
    if not os.path.exists(regression_models_path):
        raise FileNotFoundError(f"Regression models file not found: {regression_models_path}")

    with open(regression_models_path, 'r') as f:
        all_models_params = json.load(f)

    if model_type not in all_models_params:
        raise ValueError(f"Model type '{model_type}' not found.")

    model_params = all_models_params[model_type]
    slope = model_params['slope']
    intercept = model_params['intercept']

    calc_params = config.POLLUTION_CALCULATION_PARAMS

    particle_dia_microns = calc_params[model_type]['particle_diameter_microns']
    particle_density_g_cm3 = calc_params[model_type]['particle_density_g_cm3']

    particle_diameter_m = particle_dia_microns * 1e-6
    particle_density_kg_m3 = particle_density_g_cm3 * 1000

    sensor_width_m, sensor_height_m = calc_params['papersensor_size']
    area_sensor_m2 = sensor_width_m * sensor_height_m
    volume_sensor_m3 = area_sensor_m2 * sensor_width_m

    area_particle_m2 = np.pi * (particle_diameter_m / 2) ** 2
    volume_particle_m3 = (4 / 3) * np.pi * (particle_diameter_m / 2) ** 3

    num_particles_on_sensor = ((area_sensor_m2 * (analysis_results['area_percentage']/100)) / area_particle_m2
                               if area_particle_m2 > 0 else 0)

    mass_total_particles_kg = num_particles_on_sensor * volume_particle_m3 * particle_density_kg_m3
    concentration_sensor_kg_m3 = mass_total_particles_kg / volume_sensor_m3 if volume_sensor_m3 > 0 else 0
    concentration_sensor_ug_m3 = concentration_sensor_kg_m3 * 1e9
    estimated_standard_concentration_ug_m3 = intercept + slope * concentration_sensor_ug_m3

    return {
        'calculated_num_particles': num_particles_on_sensor,
        'sensor_paper_concentration_ug_m3': concentration_sensor_ug_m3,
        'estimated_standard_concentration_ug_m3': estimated_standard_concentration_ug_m3,
        'model_type_used': model_type
    }


def classify_pollution_level(concentration_ug_m3: float):
    """Clasifica nivel de contaminación basado en µg/m³"""
    if concentration_ug_m3 <= 10:
        return "Level 1 (Good): 0-10 µg/m³"
    elif concentration_ug_m3 <= 20:
        return "Level 2 (Moderate): 11-20 µg/m³"
    elif concentration_ug_m3 < 50:
        return "Level 3 (Unhealthy for Sensitive): 21-49 µg/m³"
    elif concentration_ug_m3 < 100:
        return "Level 4 (Unhealthy): 50-99 µg/m³"
    elif concentration_ug_m3 < 150:
        return "Level 5 (Very Unhealthy): 100-149 µg/m³"
    else:
        return "Level 6 (Hazardous): 150+ µg/m³"