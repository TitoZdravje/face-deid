from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CELEBA_DIR = DATA_DIR / "celeba"
CELEBA_IMAGES_DIR = CELEBA_DIR / "img_align_celeba"

IDENTITY_FILE = CELEBA_DIR / "identity_CelebA.txt"
LANDMARKS_FILE = CELEBA_DIR / "list_landmarks_align_celeba.txt"
PARTITION_FILE = CELEBA_DIR / "list_eval_partition.txt"

STICKERS_DIR = DATA_DIR / "stickers"
GENERATED_STICKERS_DIR = STICKERS_DIR / "generated"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SELECTED_SUBSETS_DIR = OUTPUTS_DIR / "selected_subsets"
SANITY_CHECKS_DIR = OUTPUTS_DIR / "sanity_checks"
STICKER_OVERLAY_CHECKS_DIR = SANITY_CHECKS_DIR / "sticker_overlay"

STICKERED_IMAGES_DIR = OUTPUTS_DIR / "stickered_images"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"

DEFAULT_SUBSET_FILE = SELECTED_SUBSETS_DIR / "celeba_subset_50_ids_seed_42.json"

CLEAN_BASELINE_RESULTS_FILE = RESULTS_DIR / "clean_baseline_facenet.csv"
CLEAN_BASELINE_PLOT_FILE = PLOTS_DIR / "clean_baseline_similarity_histogram.png"

STICKER_GRID_RESULTS_FILE = RESULTS_DIR / "sticker_grid_facenet.csv"
STICKER_GRID_SUMMARY_PLOT_FILE = (
    PLOTS_DIR / "sticker_grid_similarity_drop_by_sticker.png"
)
STICKER_GRID_POSITION_PLOT_FILE = (
    PLOTS_DIR / "sticker_grid_similarity_drop_by_position.png"
)
STICKER_GRID_SCALE_PLOT_FILE = PLOTS_DIR / "sticker_grid_similarity_drop_by_scale.png"

VERIFICATION_PAIRS_RESULTS_FILE = RESULTS_DIR / "verification_pairs_facenet.csv"
VERIFICATION_THRESHOLD_FILE = RESULTS_DIR / "verification_threshold_facenet.json"

STICKER_GRID_ANALYZED_RESULTS_FILE = RESULTS_DIR / "sticker_grid_facenet_analyzed.csv"
STICKER_GRID_ANALYSIS_SUMMARY_FILE = RESULTS_DIR / "sticker_grid_facenet_summary.csv"
STICKER_GRID_ANALYSIS_JSON_FILE = RESULTS_DIR / "sticker_grid_facenet_summary.json"

TOP_STICKER_EXAMPLES_DIR = SANITY_CHECKS_DIR / "top_sticker_examples"
