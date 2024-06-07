import os
from collections.abc import Callable, Sequence

import numpy as np

from monai.data import PatchWSIDataset
from monai.data.wsi_reader import TiffFileWSIReader
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import iter_patch_position
from monai.transforms import ForegroundMask, apply_transform
from monai.utils import convert_to_dst_type
from monai.utils.enums import CommonKeys, WSIPatchKeys


def vis_patch(image, locations, patch_size, vis_level=7):
    from monai.data import CuCIMWSIReader

    slide_reader = CuCIMWSIReader()
    slide = slide_reader.read(image)
    ratio = slide_reader.get_downsample_ratio(slide, vis_level)
    locations = np.asarray(locations)
    locations = np.divide(locations, ratio).astype(int)
    patch_size = int(patch_size // ratio)

    slide_data, _ = slide_reader.get_data(slide, level=vis_level)
    slide_data = np.moveaxis(slide_data, 0, -1)

    import cv2
    for loc in locations:
        loc = loc[::-1]
        cv2.rectangle(slide_data, tuple(loc), tuple(loc + patch_size), (0, 255, 0), 2)

    return slide_data


def vis_gt_mask(image, mask, vis_level=7):
    from monai.data import CuCIMWSIReader, TiffFileWSIReader

    slide_reader = CuCIMWSIReader(level=vis_level)
    mask_reader = TiffFileWSIReader(level=vis_level)

    slide = slide_reader.read(image)
    mask = mask_reader.read(mask)

    slide_data, _ = slide_reader.get_data(slide, level=vis_level)
    slide_data = np.moveaxis(slide_data, 0, -1)

    mask_data, _ = mask_reader.get_data(mask, level=vis_level, mode="L")
    mask_data = mask_data.squeeze()
    
    return slide_data, mask_data


# modified from https://github.com/Project-MONAI/MONAI/blob/4e0b179557f5799a4a2a9a32458dfa3afb7eb87f/monai/data/wsi_datasets.py#L321
class MaskedPatchWSIDataset(PatchWSIDataset):
    """
    This dataset extracts patches from whole slide images at the locations where foreground mask
    at a given level is non-zero. 
    Modification: 
        - The generation of patches is similar to `SlidingPatchWSIDataset`.
        - Corresponding ground truth mask is also extracted.
        - Mask reading is done by `TiffFileWSIReader` and implement in an effcient way.
        - Missing mask file is ignored, and only WSI will be patched and saved.

    Args:
        data: the list of input samples including image, location, and label (see the note below for more details).
        patch_size: the size of patch to be extracted from the whole slide image.
        patch_level: the level at which the patches to be extracted (default to 0).
        tissue_mask_level: the resolution level at which the tissue mask is created.
        threshold: the threshold for the ratio of foreground pixels to the total number of pixels in a patch.
        overlap: the overlap between patches (default to 0).
        transform: transforms to be executed on input data.
        include_label: whether to load and include labels in the output
        center_location: whether the input location information is the position of the center of the patch
        additional_meta_keys: the list of keys for items to be copied to the output metadata from the input data
        reader: the module to be used for loading whole slide imaging. Defaults to cuCIM. If `reader` is

            - a string, it defines the backend of `monai.data.WSIReader`.
            - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader,
            - an instance of a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

        kwargs: additional arguments to pass to `WSIReader` or provided whole slide reader class

    Note:
        The input data has the following form as an example:

        .. code-block:: python

            [
                {"image": "path/to/image1.tiff", "mask": "path/to/mask1.tiff"},
                {"image": "path/to/image2.tiff", "mask": "path/to/mask1.tiff", "size": [20, 20], "level": 2}
            ]

    """

    def __init__(
        self,
        data: Sequence,
        patch_size: int | tuple[int, int] | None = None,
        patch_level: int | None = None,
        tissue_mask_level: int = 7,
        threshold: float = 0.8,
        overlap: tuple[float, float] | float = 0.0,
        transform: Callable | None = None,
        include_label: bool = False,
        center_location: bool = False,
        additional_meta_keys: Sequence[str] | None = None,
        reader="cuCIM",
        **kwargs,
    ):
        super().__init__(
            data=[],
            patch_size=patch_size,
            patch_level=patch_level,
            transform=transform,
            include_label=include_label,
            center_location=center_location,
            additional_meta_keys=additional_meta_keys,
            reader=reader,
            **kwargs,
        )

        self.threshold = threshold
        self.overlap = overlap
        self.tissue_mask_level = tissue_mask_level
        self.mask_reader = TiffFileWSIReader(level=patch_level)

        # Create single sample for each patch (in a sliding window manner)
        self.data: list
        self.image_data = list(data)
        for sample in self.image_data:
            patch_samples = self._evaluate_patch_locations(sample)
            self.data.extend(patch_samples)
    
    def _get_tissue_ratio(self, mask: np.ndarray):
        return np.count_nonzero(mask) / mask.size
    
    def _get_wsi_mask_object(self, sample: dict):
        image_path = sample["mask"]
        level = self._get_level(sample)
        
        if image_path not in self.wsi_object_dict:
            if os.path.exists(image_path):
                # Save the whole image instead of TiffFile object to ensure efficient reading
                mask_data = self.mask_reader.read(image_path).asarray(level=level).astype(np.uint8)
            else:
                # If mask file is missing, use -1 to indicate that the mask is not available
                wsi_obj = self.wsi_object_dict[sample["image"]]
                h, w = self.wsi_reader.get_size(wsi_obj, level=level)
                mask_data = np.full((h, w), -1)
            if mask_data.ndim == 3:
                mask_data = mask_data[:,:,0]
            self.wsi_object_dict[image_path] = mask_data
            
        return self.wsi_object_dict[image_path]

    def _get_data(self, sample: dict):
        # Don't store OpenSlide objects to avoid issues with OpenSlide internal cache
        if self.backend == "openslide":
            self.wsi_object_dict = {}
        wsi_obj = self._get_wsi_object(sample)
        wsi_mask_obj = self._get_wsi_mask_object(sample) # (HxW)
        location = self._get_location(sample)
        level = self._get_level(sample)
        size = self._get_size(sample)

        wsi = self.wsi_reader.get_data(wsi=wsi_obj, location=location, size=size, level=level)    
        # Extract mask patch (efficient way)
        downsampling_ratio = self.wsi_reader.get_downsample_ratio(wsi=wsi_obj, level=level)
        location_ = [round(location[i] / downsampling_ratio) for i in range(len(location))]
        wsi_mask = wsi_mask_obj[location_[0] : location_[0] + size[0], location_[1] : location_[1] + size[1]]

        return wsi, wsi_mask
    
    def _transform(self, index: int):
        # Get a single entry of data
        sample: dict = self.data[index]

        # Extract patch image and associated metadata
        (image, metadata), wsi_mask = self._get_data(sample)

        # Add additional metadata from sample
        for key in self.additional_meta_keys:
            metadata[key] = sample[key]

        # Create MetaTensor output for image
        output = {CommonKeys.IMAGE: MetaTensor(image, meta=metadata), "mask": wsi_mask}

        # Include label in the output
        if self.include_label:
            output[CommonKeys.LABEL] = self._get_label(sample)

        # Apply transforms and return it
        return apply_transform(self.transform, output) if self.transform else output
    
    def _evaluate_patch_locations(self, sample):
        """Calculate the location for each patch based on the mask at different resolution level"""
        patch_size = self._get_size(sample)
        patch_level = self._get_level(sample)
        wsi_obj = self._get_wsi_object(sample)

        # load the entire image at level=tissue_mask_level
        wsi, _ = self.wsi_reader.get_data(wsi_obj, level=self.tissue_mask_level)
        wsi_size = self.wsi_reader.get_size(wsi_obj, 0)

        # create the foreground tissue mask and get all indices for non-zero pixels
        tissue_mask = np.squeeze(convert_to_dst_type(ForegroundMask(hsv_threshold={"S": "otsu"})(wsi), dst=wsi)[0])
        tissue_mask_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, self.tissue_mask_level)

        # generate patch locations at level 0 in sliding window manner
        patch_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, patch_level)
        patch_size_0 = np.array([p * patch_ratio for p in patch_size])  # patch size at level 0
        patch_locations = np.array(
            list(
                iter_patch_position(
                    image_size=wsi_size, patch_size=patch_size_0, overlap=self.overlap, padded=False
                )
            )
        )

        # filter out patches that are outside of the tissue mask
        patch_locations_filtered = []
        tissue_mask_patch_size = np.divide(patch_size_0, tissue_mask_ratio).astype(int)
        for loc in patch_locations:
            tissue_mask_loc = np.divide(loc, tissue_mask_ratio).astype(int)
            tissue_ratio = self._get_tissue_ratio(
                tissue_mask[
                    tissue_mask_loc[0] : tissue_mask_loc[0] + tissue_mask_patch_size[0],
                    tissue_mask_loc[1] : tissue_mask_loc[1] + tissue_mask_patch_size[1],
                ]
            )
            if tissue_ratio >= self.threshold:
                patch_locations_filtered.append(loc)
        patch_locations = np.asarray(patch_locations_filtered)

        # fill out samples with location and metadata
        sample[WSIPatchKeys.SIZE.value] = patch_size
        sample[WSIPatchKeys.LEVEL.value] = patch_level
        return [
            {**sample, WSIPatchKeys.LOCATION.value: np.array(loc)} for loc in patch_locations
        ]