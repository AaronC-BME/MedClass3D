"""3D Grad-CAM / Layer-CAM visualization for MedClass3D classifiers.

Ported from SSL3D_classification's gradcam suite. The core compute/rendering
modules (``gradcam3d_viz``, ``multimodal_gradcam``) are model-agnostic; the
MedClass3D-specific loading, patient assembly, and per-fold drivers live in
``runner``.

monai is required only for the occlusion / true-/guided-Grad-CAM methods and is
imported lazily, so Layer-CAM / NotGradCAM / Integrated-Gradients work without it.
"""

from .gradcam3d_viz import GradcamConfig, run_gradcam, ALL_METHODS
from .multimodal_gradcam import MultiModalConfig, run_multimodal_gradcam
from .runner import (
    get_stages_for_model,
    find_best_ckpt_for_fold,
    derive_method_subdir,
    reorganize_to_per_image_subdirs,
    load_fold_model,
    class_names_for,
    build_testset_patients,
    build_external_patients,
    bucket_patients_by_confusion_cell,
    print_confusion_bucket_summary,
    run_gradcam_for_patients,
)
from .averaging import average_layercam

__all__ = [
    "GradcamConfig",
    "run_gradcam",
    "ALL_METHODS",
    "MultiModalConfig",
    "run_multimodal_gradcam",
    "get_stages_for_model",
    "find_best_ckpt_for_fold",
    "derive_method_subdir",
    "reorganize_to_per_image_subdirs",
    "load_fold_model",
    "class_names_for",
    "build_testset_patients",
    "build_external_patients",
    "bucket_patients_by_confusion_cell",
    "print_confusion_bucket_summary",
    "run_gradcam_for_patients",
    "average_layercam",
]
