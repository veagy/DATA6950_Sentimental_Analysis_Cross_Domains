"""
Data Package Module exports
"""

# Native Phase 1 Exports 
from .data_loader import any_to_loader, make_loader
from .data_source import load_dataframe, make_loader_from_source

# Phase 2 Exports (Cleaning Scripts)
from .clean_text import clean_text, clean_code
from .clean_tabular import clean_tabular
from .clean_image import denoise_image, median_filter_tensor
from .clean_audio import butter_filter, noise_gate
from .clean_outliers import winsorise, log_transform_positive, mark_outliers_as_nan
from .pipeline import TabularPipeline

# Phase 3 Exports (Feature Engineering & Scaling Pipelines)
from .feature_engineering import ratio_features, lag_features, rolling_mean, pairwise_interactions
from .serialization import save_preprocessor, load_preprocessor
from .scaling import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
    Binarizer,
    PowerTransformer,
    QuantileTransformer,
)
from .feature_engineering_transforms import (
    PolynomialFeatures,
    SplineTransformer,
    FunctionTransformer,
    KernelCenterer,
)
from .pipeline_core import Pipeline
from .outlier_detection import OneClassSVM

# Phase 4 Exports (Modality-Specific Preprocessing)
from .nlp_preprocessing import (
    classical_nlp_pipeline,
    get_huggingface_tokenizer,
    tokenize_texts,
    mask_prompt_tokens,
    SFTCollator,
)
from .image_preprocessing import (
    get_train_transform,
    get_val_transform,
    compute_mean_std,
    mixup,
    cutmix,
    extract_patches,
    normalize_multichannel,
)
from .audio_preprocessing import (
    load_canonical_audio,
    get_mfcc_extractor,
    extract_delta_features,
    get_log_mel_spectrogram_extractor,
    get_stft_extractor,
    specaugment,
    augment_audio_waveform,
)
from .video_preprocessing import (
    load_video_tensor,
    spatial_video_transform,
    compute_dense_optical_flow,
    build_clips,
    VideoClipDataset,
)

# Phase 5 Exports (Paradigm-Specific AI Preprocessing)
from .paradigm_supervised import (
    build_weighted_sampler,
    compute_class_weights,
    apply_smote,
    stratified_split,
    transform_heavy_tail_target,
)
from .paradigm_unsupervised import (
    normalize_l2,
    UnsupervisedAutoencoderPipeline,
)
from .paradigm_semi_supervised import (
    ContrastiveDataset,
    contrastive_loss_ntxent,
    consistency_loss,
)
from .paradigm_rl import (
    RunningNormalizer,
    RewardScaler,
    ReplayBuffer,
    FrameStack,
)
from .paradigm_generative import (
    pack_documents,
    quality_filter,
    get_minhash_dedup,
    get_noise_schedule,
    add_noise,
    gradient_penalty,
)
from .embeddings_clustering import (
    l2_normalize,
    generate_cosine_pseudo_labels,
    apply_pca_reduction,
    apply_umap_reduction,
    apply_hdbscan_clustering,
    class_tfidf_labels,
)

# Phase 6 Exports (Online Learning and Streaming Inference)
from .streaming_ingestion import (
    tail_file,
    AsyncioSensorReader,
    get_kafka_consumer_stream,
    get_websocket_stream,
)
from .online_scalers import (
    OnlineStandardScaler,
    EMAScaler,
    SlidingWindowScaler,
    OnlineLabelEncoder,
)
from .drift_detection import (
    PageHinkley,
    ADWIN,
    DDM,
)
from .incremental_learning import (
    EWCPenalty,
    ExperienceReplay,
    online_sgd_step,
    warm_start_update,
)
from .streaming_nlp_asr import (
    classify_text_online,
    StreamingASRPipeline,
)
from ..models.machine_learning.preprocessing.imputation.imputation import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer,
    MissingIndicator,
)
from ..models.machine_learning.preprocessing.encoding.encoding import (
    ContinuousFeatureDiscretizer,
    LabelEncoder,
    LabelBinarizer,
    MultiLabelBinarizer,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    KBinsDiscretizer,
)
from ..models.machine_learning.preprocessing.outlier.outlier import (
    IsolationForest,
    LocalOutlierFactor,
)

__all__ = [
    # Data factories
    "any_to_loader",
    "make_loader",
    "load_dataframe",
    "make_loader_from_source",

    # Cleaning
    "clean_text",
    "clean_code",
    "clean_tabular",
    "denoise_image",
    "median_filter_tensor",
    "butter_filter",
    "noise_gate",
    "winsorise",
    "log_transform_positive",
    "mark_outliers_as_nan",
    "TabularPipeline",

    # Phase 3 Feature Engineering
    "ratio_features",
    "lag_features",
    "rolling_mean",
    "pairwise_interactions",
    "save_preprocessor",
    "load_preprocessor",
    
    # Phase 3 Scalers & Feature Transformers
    "StandardScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "Normalizer",
    "Binarizer",
    "PowerTransformer",
    "QuantileTransformer",
    "PolynomialFeatures",
    "QuantileTransformer",
    "PolynomialFeatures",
    "SplineTransformer",
    "FunctionTransformer",
    "KernelCenterer",
    "Pipeline",
    "Pipeline",

    # Imputation
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "MissingIndicator",

    # Encoding
    "ContinuousFeatureDiscretizer",
    "LabelEncoder",
    "LabelBinarizer",
    "MultiLabelBinarizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "KBinsDiscretizer",

    # Outlier detection
    "OneClassSVM",
    "IsolationForest",
    "LocalOutlierFactor",

    # Phase 4 NLP Preprocessing
    "classical_nlp_pipeline",
    "get_huggingface_tokenizer",
    "tokenize_texts",
    "mask_prompt_tokens",
    "SFTCollator",

    # Phase 4 Image Preprocessing
    "get_train_transform",
    "get_val_transform",
    "compute_mean_std",
    "mixup",
    "cutmix",
    "extract_patches",
    "normalize_multichannel",

    # Phase 4 Audio Preprocessing
    "load_canonical_audio",
    "get_mfcc_extractor",
    "extract_delta_features",
    "get_log_mel_spectrogram_extractor",
    "get_stft_extractor",
    "specaugment",
    "augment_audio_waveform",

    # Phase 4 Video Preprocessing
    "load_video_tensor",
    "spatial_video_transform",
    "compute_dense_optical_flow",
    "build_clips",
    "VideoClipDataset",

    # Phase 5 Supervised
    "build_weighted_sampler",
    "compute_class_weights",
    "apply_smote",
    "stratified_split",
    "transform_heavy_tail_target",

    # Phase 5 Unsupervised
    "normalize_l2",
    "UnsupervisedAutoencoderPipeline",

    # Phase 5 Semi-Supervised
    "ContrastiveDataset",
    "contrastive_loss_ntxent",
    "consistency_loss",

    # Phase 5 Reinforcement Learning
    "RunningNormalizer",
    "RewardScaler",
    "ReplayBuffer",
    "FrameStack",

    # Phase 5 Generative Models
    "pack_documents",
    "quality_filter",
    "get_minhash_dedup",
    "get_noise_schedule",
    "add_noise",
    "gradient_penalty",

    # Phase 5 Embeddings & Clustering
    "l2_normalize",
    "generate_cosine_pseudo_labels",
    "apply_pca_reduction",
    "apply_umap_reduction",
    "apply_hdbscan_clustering",
    "class_tfidf_labels",

    # Phase 6 Streaming Ingestion
    "tail_file",
    "AsyncioSensorReader",
    "get_kafka_consumer_stream",
    "get_websocket_stream",

    # Phase 6 Online Scalers
    "OnlineStandardScaler",
    "EMAScaler",
    "SlidingWindowScaler",
    "OnlineLabelEncoder",

    # Phase 6 Drift Detection
    "PageHinkley",
    "ADWIN",
    "DDM",

    # Phase 6 Incremental Learning
    "EWCPenalty",
    "ExperienceReplay",
    "online_sgd_step",
    "warm_start_update",

    # Phase 6 Streaming NLP/ASR
    "classify_text_online",
    "StreamingASRPipeline",
]
