import numbers
from copy import deepcopy
from typing import List, Tuple

import cv2
import numpy as np
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from torch import Tensor
from torchvision.ops.boxes import box_area
from torchvision.transforms.functional import crop, get_image_size, pad


class ResizeShortestSide(ResizeLongestSide):
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / min(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def boxes_to_circles(boxes, height, width):
    boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], 0, width - 1)
    boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], 0, height - 1)
    src_widths = boxes[:, 2] - boxes[:, 0]
    src_heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * src_widths
    ctr_y = boxes[:, 1] + 0.5 * src_heights
    return torch.stack([ctr_x, ctr_y], dim=1), torch.stack([src_widths, src_heights], dim=1) / 2


def expand_boxes(boxes: torch.tensor, ratio, height, width, symmetry=False, half_symmetry=False, square=False):
    if square:
        boxes_width = boxes[:, 2] - boxes[:, 0]
        boxes_height = boxes[:, 3] - boxes[:, 1]
        largest_size = torch.stack([boxes_height, boxes_width], 1).max(1)[0]
        pad_width = (largest_size - boxes_width) / 2
        pad_height = (largest_size - boxes_height) / 2
        boxes[:, 0] = torch.clamp(boxes[:, 0] - pad_width, 0, width)
        boxes[:, 2] = torch.clamp(boxes[:, 2] + pad_width, 0, width)
        boxes[:, 1] = torch.clamp(boxes[:, 1] - pad_height, 0, height)
        boxes[:, 3] = torch.clamp(boxes[:, 3] + pad_height, 0, height)

    if symmetry:
        pad_w = torch.stack([ratio * width + (boxes[:, 0] * 0),
                             boxes[:, 0], width - 1 - boxes[:, 2]], 1).min(1)[0]
        pad_h = torch.stack([ratio * height + (boxes[:, 0] * 0),
                             boxes[:, 1], height - 1 - boxes[:, 3]], 1).min(1)[0]
        boxes[:, 0] = boxes[:, 0] - pad_w
        boxes[:, 2] = boxes[:, 2] + pad_w
        boxes[:, 1] = boxes[:, 1] - pad_h
        boxes[:, 3] = boxes[:, 3] + pad_h
    elif half_symmetry:
        pad_l = torch.stack([ratio * width + (boxes[:, 0] * 0), boxes[:, 0]], 1).min(1)[0]
        pad_r = torch.stack([ratio * width + (boxes[:, 0] * 0), width - 1 - boxes[:, 2]], 1).min(1)[0]
        pad_t = torch.stack([ratio * height + (boxes[:, 0] * 0), boxes[:, 1]], 1).min(1)[0]
        pad_b = torch.stack([ratio * height + (boxes[:, 0] * 0), height - 1 - boxes[:, 3]], 1).min(1)[0]
        if height < width:
            pad_l = pad_r = torch.stack([pad_l, pad_r], 1).min(1)[0]
        else:
            pad_t = pad_b = torch.stack([pad_t, pad_b], 1).min(1)[0]
        boxes[:, 0] = boxes[:, 0] - pad_l
        boxes[:, 2] = boxes[:, 2] + pad_r
        boxes[:, 1] = boxes[:, 1] - pad_t
        boxes[:, 3] = boxes[:, 3] + pad_b
    else:
        boxes[:, 0] = torch.clamp(boxes[:, 0] - ratio * width, 0, width - 1)
        boxes[:, 2] = torch.clamp(boxes[:, 2] + ratio * width, 0, width - 1)
        boxes[:, 1] = torch.clamp(boxes[:, 1] - ratio * height, 0, height - 1)
        boxes[:, 3] = torch.clamp(boxes[:, 3] + ratio * height, 0, height - 1)
    return boxes


class CropImages:
    def __init__(self, crop_box: torch.tensor) -> None:
        self.crop_box = crop_box

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Crops a numpy array with shape HxWxC in uint8 format.
        """
        x1, y1, x2, y2 = self.crop_box.int()
        return image[y1:y2, x1:x2]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension.
        """
        x1, y1, x2, y2 = self.crop_box.int().numpy()
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] - x1
        coords[..., 1] = coords[..., 1] - y1
        return coords

    def apply_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array shape Bx4.
        """
        lt = boxes[:, :2] - self.crop_box[None, :2]
        rb = self.crop_box[None, -2:] - boxes[:, -2:]
        ltrb = np.concatenate([lt, rb], 1)
        inside_inds = np.nonzero((ltrb > 0).all(1))[0]

        boxes = self.apply_coords(boxes.reshape(-1, 2, 2))
        return boxes.reshape(-1, 4), inside_inds

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        x1, y1, x2, y2 = self.crop_box.int()
        return image[..., y1:y2, x1:x2]

    def apply_coords_torch(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension.
        """
        x1, y1, x2, y2 = self.crop_box.int()
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] - x1
        coords[..., 1] = coords[..., 1] - y1
        return coords

    def apply_boxes_torch(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4.
        """
        lt = boxes[:, :2] - self.crop_box[None, :2]
        rb = self.crop_box[None, -2:] - boxes[:, -2:]
        ltrb = torch.cat([lt, rb], 1)
        inside_inds = torch.nonzero((ltrb > 0).all(1), as_tuple=True)[0]

        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2))
        return boxes.reshape(-1, 4), inside_inds

    def apply_masks_torch(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape BxHxW.
        """
        x1, y1, x2, y2 = self.crop_box.int()
        return masks[..., y1:y2, x1:x2]


def bias_crop(img: Tensor, output_size: List[int], box: Tensor) -> Tensor:
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    image_width, image_height = get_image_size(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
        image_width, image_height = get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    l = cx - crop_width * 0.5
    r = cx + crop_width * 0.5
    t = cy - crop_height * 0.5
    b = cy + crop_height * 0.5

    if l < 0:
        crop_left = 0
    elif r >= image_width:
        crop_left = image_width - crop_width
    else:
        crop_left = l
    if t < 0:
        crop_top = 0
    elif b >= image_height:
        crop_top = image_height - crop_height
    else:
        crop_top = t
    crop_top = int(crop_top)
    crop_left = int(crop_left)
    # crop_top = int(round((image_height - crop_height) / 2.0))
    # crop_left = int(round((image_width - crop_width) / 2.0))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def compute_bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    # both boxes: xyxy
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = (inter + 1e-6) / (union + 1e-6)
    return iou, inter, union


def str2rgb(color, alpha=None):
    if color == 'red':
        rgb = (255, 0, 0)
    elif color == 'cpt_red':
        rgb = (240, 0, 30)
    elif color == 'green':
        rgb = (0, 255, 0)
    elif color == 'blue':
        rgb = (0, 0, 255)
    elif color == 'purple':
        rgb = (128, 0, 128)
    elif color == 'yellow':
        rgb = (255, 255, 0)
    elif color == 'cyan':
        rgb = (0, 255, 255)
    elif color == 'black':
        rgb = (0, 0, 0)
    elif color == 'white':
        rgb = (255, 255, 255)
    elif color == 'grey':
        rgb = (124, 116, 104)
    else:
        raise NotImplementedError

    if alpha is None:
        return rgb
    rgba = (*rgb, alpha)
    return rgba


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return cnt
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def scale_mask(mask, scale):
    if scale == 1:
        return mask

    contours, hierarchy = cv2.findContours(mask.astype(
        np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    scaled_contours = [scale_contour(cnt, scale) for cnt in contours]

    mask_scaled = np.zeros(mask.shape, np.uint8)
    mask_scaled = cv2.drawContours(mask_scaled, scaled_contours, -1, (255), -1)
    return mask_scaled > 0
