from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, target_dtype: Type[number] = np.float32):
        self.use_mask_for_norm = None
        self.intensityproperties = None
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype, copy=False)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image -= mean
            image /= (max(std, 1e-8))
        return image


class CTNormalization(ImageNormalization):
    """
    nnU-Net-style CT normalization.

    Requires `self.intensityproperties` to be set before calling `.run()`.
    The dict must contain:
        mean              : float, dataset-wide foreground mean
        std               : float, dataset-wide foreground std
        percentile_00_5   : float, lower clip value (0.5th percentile)
        percentile_99_5   : float, upper clip value (99.5th percentile)

    The normalization:
        1. Clips voxel values to [percentile_00_5, percentile_99_5].
        2. Subtracts the dataset-wide mean and divides by the dataset-wide std.

    Unlike ZScoreNormalization, the statistics here come from the *whole dataset*,
    not the per-image distribution. This preserves the absolute meaning of HU
    values across cases, which matters for CT where intensity is calibrated.
    """
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, (
            "CTNormalization requires `intensityproperties` to be set before calling .run(). "
            "Set normalizer.intensityproperties = {'mean': ..., 'std': ..., "
            "'percentile_00_5': ..., 'percentile_99_5': ...} first."
        )
        required_keys = {"mean", "std", "percentile_00_5", "percentile_99_5"}
        missing = required_keys - set(self.intensityproperties.keys())
        assert not missing, f"intensityproperties is missing keys: {missing}"

        image = image.astype(self.target_dtype, copy=False)
        lower = self.intensityproperties["percentile_00_5"]
        upper = self.intensityproperties["percentile_99_5"]
        mean = self.intensityproperties["mean"]
        std = self.intensityproperties["std"]

        np.clip(image, lower, upper, out=image)
        image -= mean
        image /= max(std, 1e-8)
        return image