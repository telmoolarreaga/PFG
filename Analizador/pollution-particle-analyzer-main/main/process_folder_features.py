import os
import argparse
import pandas as pd
from multiprocessing import Pool, cpu_count

from main.roi import roi_extraction
from main.preprocessing import run_preprocessing_pipeline
from main.analysis import analyze_particles, extract_features_per_particle
from main import config


# ---------------------------------------------------------
# Procesar una sola imagen (worker)
# ---------------------------------------------------------

def process_single_image(args):

    img_name, input_folder, temp_root = args
    img_path = os.path.join(input_folder, img_name)

    try:
        # carpeta única para esta imagen (evita colisiones)
        process_dir = os.path.join(temp_root, os.path.splitext(img_name)[0])
        os.makedirs(process_dir, exist_ok=True)

        # 1️⃣ ROI
        roi_path = roi_extraction(img_path, process_dir)
        if not roi_path:
            return None

        # 2️⃣ preprocessing
        binary_mask_path, grayscale_roi_path = run_preprocessing_pipeline(
            roi_path,
            process_dir
        )

        # 3️⃣ análisis de partículas
        analysis_results = analyze_particles(
            binary_image_path=binary_mask_path,
            original_image_path=grayscale_roi_path,
            output_dir=None,
            filter_params=config.DEFAULT_FILTER_PARAMETERS
        )

        filtered_regions = analysis_results["filtered_regions"]
        if len(filtered_regions) == 0:
            return None

        # 4️⃣ features
        df_particles = extract_features_per_particle(
            filtered_regions,
            original_img=analysis_results.get("original_img", None),
            min_area=config.DEFAULT_FILTER_PARAMETERS.get("min_area", 0)
        )

        if df_particles.empty:
            return None

        df_particles.insert(0, "image_name", img_name)

        return df_particles

    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        return None


# ---------------------------------------------------------
# Guardar batch en CSV
# ---------------------------------------------------------

def save_batch(batch_results, csv_path):

    if not batch_results:
        return

    df_batch = pd.concat(batch_results, ignore_index=True)

    if os.path.exists(csv_path):
        df_batch.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_batch.to_csv(csv_path, index=False)


# ---------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------

def process_folder(input_folder, output_dir, model):

    os.makedirs(output_dir, exist_ok=True)

    temp_root = os.path.join(output_dir, "temp_processing")
    os.makedirs(temp_root, exist_ok=True)

    csv_path = os.path.join(output_dir, "particle_features.csv")

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]

    print(f"Found {len(images)} images")

    # ---------------------------------------------------------
    # reanudación
    # ---------------------------------------------------------

    processed_images = set()

    if os.path.exists(csv_path):

        df_existing = pd.read_csv(csv_path)

        if "image_name" in df_existing.columns:
            processed_images = set(df_existing["image_name"].unique())

        print(f"Resuming: {len(processed_images)} images already processed")

    images = [img for img in images if img not in processed_images]

    print(f"{len(images)} images remaining")

    if not images:
        print("All images already processed")
        return

    # ---------------------------------------------------------
    # multiprocessing
    # ---------------------------------------------------------

    num_workers = max(cpu_count() - 1, 1)
    print(f"Using {num_workers} CPU cores")

    args_list = [(img, input_folder, temp_root) for img in images]

    save_interval = 100
    processed_count = 0
    batch_results = []

    with Pool(num_workers) as pool:

        for result in pool.imap_unordered(process_single_image, args_list):

            if result is not None:
                batch_results.append(result)

            processed_count += 1

            if processed_count % save_interval == 0:

                save_batch(batch_results, csv_path)

                print(f"Checkpoint saved at {processed_count} images")

                batch_results = []

    # guardar resto
    if batch_results:
        save_batch(batch_results, csv_path)
        print("Final batch saved")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process folder of images and extract particle features"
    )

    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder with images"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="PM25",
        choices=["PM10", "PM25"],
        help="Pollution model type"
    )

    args = parser.parse_args()

    process_folder(
        args.input_folder,
        args.output_dir,
        args.model
    )