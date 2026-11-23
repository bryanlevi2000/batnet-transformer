

# ============================================================
# Global settings
# ============================================================

INPUT_DIR = r"C:\Users\Students\Box\BatNET 2024-2026\Bat Recordings (Call Library)\Subsets for Development\model_training\111326V1"
NUM_WORKERS = 16

# ============================================================
# Stage 1: ZC Extraction
# ============================================================

OUTPUT_DIR = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111326V1\2_Classifier"

# Output mode - specify "individual" to have one CSV per pulse,
# or "combined" to have one CSV per WAV file
OUTPUT_MODE = "individual"  # individual or combined

# --------------------- Denoising ---------------------

NOISE_FACTOR = 2.0
NOISE_STRIDE = 4

# --------------------- ZC Extraction ---------------------

DIVRAT = 8
HYSTERESIS_PERCENTILE = 85.0

# --------------------- Pulse Detection ---------------------

# Local linearity
LOOKAHEAD_N = 15
LINEARITY_CUTOFF = 0.9

# Greedy walk
TIME_CUTOFF = 500  # microseconds
INC_FREQ_CUTOFF_HZ = 1000
DEC_FREQ_CUTOFF_HZ = 5000
MIN_POINTS = 15
BUFFER_US = 10000

# --------------------- Pulse Cleanup ---------------------

# Tail removal
TAIL_TIME_FRACTION = 0.1
TAIL_DEC_FREQ_CUTOFF_HZ = 1000

# Pulse metrics
MIN_BANDWIDTH_HZ = 1000
MAX_BANDWIDTH_HZ = 120000

MIN_DURATION_US = 100
MAX_DURATION_US = 30000

FMIN_RANGES = (5000, 50000)
FMAX_RANGES = (10000, 130000)

# --------------------- Pulse Metrics ---------------------

METRIC_FC_PROPORTION = 0.33 # Fc can only occur in this last proportion of the call in time
METRIC_KNEE_PROPORTION = 0.1 # Knee cannot occur in this first proportion of the call in time
METRIC_OUTPUT_FILE = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111326V1\2_Classifier\metrics_output.csv"

# ============================================================
# Stage 2: Preprocessing
# ============================================================

PULSE_FILTERED_REJECTION_FILE = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111326V1\2_Classifier\pulse_rejection_summary.csv"
PCHIP_TIME_INTERVAL = 100  # microseconds

# Currently, the possible features we can include are freq_hz, amplitude, slope, position
NORMALIZATION_COLUMNS = [
    "freq_hz",
    "amplitude",
    "slope",
    "position"
]

# ============================================================
# Stage 3: Shard creation
# ============================================================

SHARD_NUM_SAMPLES = 50000 # Maximum number of samples per shard
SHARD_PARENT_DIR = r"C:\Users\Students\Desktop\Working\model_training_111326V1\shards"
VAL_RATIO = 0.15

# ============================================================
# Stage 4: Transformer (mvts_transformer, Zerveas et al., KDD 2021)
# ============================================================

# --------------------- Data ---------------------

# Both folders are stage 3 shard folders and need train and validation splits (VAL_RATIO > 0).
PRETRAIN_SHARD_DIR = r"C:\Users\Students\Desktop\Working\model_training_111326V1\shards" # Encoder pretraining; labels are ignored
FINETUNE_SHARD_DIR = r"C:\Users\Students\Desktop\Working\model_training_111326V1\shards" # Classification fine-tuning
TRANSFORMER_OUTPUT_DIR = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111626V1\2_Classifier" # Run folders are written here

FEATURE_COLUMNS = ["freq_hz", "amplitude", "slope"] # Subset of NORMALIZATION_COLUMNS the model sees
MAX_SEQUENCE_LENGTH = 200 # Every sample is padded (or truncated) to this length
RAM_RESERVE_GB = 4

# --------------------- Model ---------------------

# Pretraining and fine-tuning must use the same model settings and FEATURE_COLUMNS.
D_MODEL = 64
NUM_HEADS = 8
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
POS_ENCODING = "learnable" # "learnable" or "fixed" (sinusoidal)
ACTIVATION = "gelu" # "gelu" or "relu"
NORMALIZATION_LAYER = "BatchNorm" # "BatchNorm" (paper) or "LayerNorm"
CLASSIFIER_HEAD = "flatten" # "flatten" (paper), "mean" (masked mean pooling) or "attention" (masked attention pooling)

# --------------------- Training (pretraining and fine-tuning) ---------------------

BATCH_SIZE = 64
OPTIMIZER = "RAdam" # "Adam", "AdamW" or "RAdam"
LEARNING_RATE = 0.001

# If you want decay:
# LR_STEPS = [100, 200]
# LR_FACTORS = [0.1, 0.5]
# Would give 
# Epochs 1 to 100: 0.001
# Epochs 101 to 200: 0.0001 (0.001 × 0.1)
# Epochs 201 onward: 0.00005 (0.0001 × 0.5)

LR_STEPS = [] # Epochs after which the learning rate is multiplied by the matching LR_FACTORS entry
LR_FACTORS = [] # Same length as LR_STEPS, e.g. LR_STEPS = [100, 200] and LR_FACTORS = [0.1, 0.1]
L2_REG = 0.0 # L2 penalty weight
GLOBAL_REG = False # True: L2_REG is optimizer weight decay on all weights; False: L2 on the output layer only
GRAD_CLIP_NORM = 4.0 # Max total gradient norm, as in mvts_transformer
VAL_INTERVAL = 2 # Validate every N epochs (also before training, after the first epoch, and after the last)
SEED = 1 # None disables seeding; set per epoch as SEED + epoch
SAVE_EVERY_EPOCH = False # Also keep epochs/epoch_XXX.pt for every epoch (last_model.pt is always written)
RESUME_FROM = None # last_model.pt or epochs/epoch_XXX.pt of a run of the same script; None starts fresh

# --------------------- Pretraining (transformer/pretrain.py) ---------------------

PRETRAIN_EPOCHS = 200
MASKING_RATIO = 0.15 # Share of each feature's time steps that is masked and reconstructed
MEAN_MASK_LENGTH = 3 # Mean length of a masked segment, in time steps

# --------------------- Fine-tuning (transformer/finetune.py) ---------------------

FINETUNE_EPOCHS = 100
PRETRAINED_ENCODER_PATH = None # best_model.pt of a pretraining run; None trains the classifier from scratch
FREEZE_ENCODER = False # True trains only the classifier head
KEY_METRIC = "accuracy" # Validation metric that picks best_model.pt: "loss", "accuracy" or "precision"

# --------------------- Inference ---------------------

TEST_DIR = r"C:\Users\Students\Downloads\usfws_13" # WAVs to predict; each WAV's parent folder is its ground truth
CHECKPOINT_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111626V1\2_Classifier\finetune_20261117_172333\best_model.pt" # best_model.pt of a fine-tuning run
PREDICTIONS_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111626V2\4_Outputs\predictions.csv"
FAILED_FILES_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111626V2\4_Outputs\failed_files.csv" # WAVs that could not be processed

# validation_inference.py: validation WAVs listed in FINETUNE_SHARD_DIR, found under INPUT_DIR
VAL_PREDICTIONS_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111626V2\4_Outputs\val_predictions.csv"
VAL_FAILED_FILES_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111626V2\4_Outputs\val_failed_files.csv"

