import argparse
import json
import os
from typing import List

import clip
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from segment_anything import sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from segment_anything.utils.amg import (batched_mask_to_box,
                                        remove_small_regions)
from segment_anything.utils.transforms import ResizeLongestSide
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from fine_grained_visual_prompt import FGVP_ENSEMBLE


class ClipModel(nn.Module):
    def __init__(self, model: nn.Module, tokenizer, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def forward(self, image: torch.Tensor, text: List[str], softmax=False):
        text = [t.lower() for t in text]
        tokenized_text = self.tokenizer(text).to(self.device)
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(tokenized_text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.t()  # N, M (0~1)
        if softmax:
            similarity = (100 * similarity).softmax(-1)
        return similarity


def draw_box(image, bbox, color):
    # color: bgr
    # bbox: [x1, y1, x2, y2]
    thickness = 2

    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=thickness)
    return image


def draw_mask(image, mask, color):
    # color: bgr
    # mask: [h, w]
    alpha = 0.3
    draw_contours = True
    coutour_thickness = 2

    mask = mask > 0
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    overlay = image.copy()
    overlay[mask] = color
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0.0)
    if draw_contours:
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), coutour_thickness)

    return image


def draw_text(image, text, left, top, color, as_center=False):
    # color: bgr
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    font_scale = 0.8
    thickness = 1
    bkg_alpha = 0.6
    top_relaxation = 8
    bottom_relaxation = 2

    # calculate text width and height
    retval, baseLine = cv2.getTextSize(text, fontFace=font_face, fontScale=font_scale, thickness=thickness)

    # calculate text location
    top = max(top, retval[1] + baseLine + top_relaxation)
    lt = [left, top - retval[1] - baseLine - top_relaxation]
    rb = [left + retval[0], top - bottom_relaxation]
    text_lt = [left, top - baseLine]

    if as_center:
        shift_x = (rb[0] - lt[0]) // 2
        shift_y = (rb[1] - lt[1]) // 2
        lt[0] -= shift_x
        rb[0] -= shift_x
        lt[1] -= shift_y
        rb[1] -= shift_y
        text_lt = [left - shift_x, top - baseLine - shift_y]

    overlay = image.copy()
    overlay = cv2.rectangle(overlay, lt, rb, thickness=-1, color=[0, 0, 0])
    image = cv2.addWeighted(overlay, bkg_alpha, image, 1 - bkg_alpha, 0.0)
    image = cv2.putText(image, text, text_lt, fontScale=font_scale, fontFace=font_face,
                        color=color, thickness=thickness, lineType=line_type)
    return image


def get_masks(args, image):
    if args.sam_prompt == "box" or args.sam_prompt == "keypoint":
        ori_size = image.shape[:2]
        image = resize_transform_sam.apply_image(image)
        new_size = image.shape[:2]

        sam_inputs = torch.as_tensor(image, device=device)
        sam_inputs = sam_inputs.permute(2, 0, 1).contiguous()[None, :, :, :]
        sam_inputs = sam_model.preprocess(sam_inputs)

        if args.sam_prompt == "keypoint":
            with open(args.candidate_points, 'r') as f:
                points = json.load(f)
            in_points = resize_transform_sam.apply_coords_torch(points, ori_size)
            in_points = in_points.to(device)
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=device)
            in_points = (in_points[:, None, :], in_labels[:, None])
        else:
            in_points = None

        if args.sam_prompt == "box":
            with open(args.candidate_boxes, 'r') as f:
                boxes = torch.from_numpy(np.array(json.load(f)))
            in_boxes = resize_transform_sam.apply_boxes_torch(boxes, ori_size)
            in_boxes = in_boxes[:, None, :].to(device)
        else:
            in_boxes = None

        features = sam_model.image_encoder(sam_inputs)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=in_points,
            boxes=in_boxes,
            masks=None,
        )
        low_res_masks, iou_pred = sam_model.mask_decoder(
            image_embeddings=features,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.sam_multimask_output,
        )
        masks = F.interpolate(low_res_masks, (args.sam_image_size, args.sam_image_size),
                              mode="bilinear", align_corners=False)  # N, 1, H, W
        masks = masks[:, :1, :new_size[0], :new_size[1]]
        masks = F.interpolate(masks, ori_size, mode="bilinear", align_corners=False)
        masks = masks > sam_model.mask_threshold

        if args.min_mask_region_area > 0:
            bit_masks = masks > sam_model.mask_threshold
            masks = []
            for mask in bit_masks:
                mask = mask.squeeze(0).cpu().numpy()
                mask, changed = remove_small_regions(mask, args.min_mask_region_area, mode="holes")
                mask, changed = remove_small_regions(mask, args.min_mask_region_area, mode="islands")
                mask = torch.as_tensor(mask, device=device).unsqueeze(0)
                masks.append(mask)
            masks = torch.stack(masks, 0)
        if args.recompute_box:
            masks = masks > sam_model.mask_threshold
            boxes = batched_mask_to_box(bit_masks.squeeze(1)).float()
    else:
        assert args.sam_prompt == 'grid'
        outputs = sam_mask_generator.generate(image)
        masks = torch.from_numpy(np.stack([x['segmentation'] for x in outputs])).unsqueeze(1)
        boxes = torch.from_numpy(np.stack([x['bbox'] for x in outputs])).float()
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

    return boxes, masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', metavar='DIR', required=True,
                        help='path to dataset (root dir)')
    parser.add_argument('--text', nargs="+", required=True,
                        help='text to be located')
    parser.add_argument('--candidate_boxes', default=None, type=str,
                        help='text to be located')
    parser.add_argument('--candidate_points', default=None, type=str,
                        help='text to be located')
    parser.add_argument('--out_dir', type=str, default='./demo',
                        help='text to be located')
    # prompt args
    parser.add_argument('--visual_prompt', nargs='+', default=['blur_mask'],
                        choices=['none',
                                 'crop', 'cpt', 'cpt_seg', 'ov_seg', 'blur_seg',
                                 'keypoint',
                                 'box', 'box_mask', 'blur_box_mask', 'reverse_box_mask', 'grayscale_box_mask',
                                 'circle', 'circle_mask', 'blur_circle_mask', 'reverse_circle_mask', 'grayscale_circle_mask',
                                 'contour', 'mask', 'blur_mask', 'reverse_mask', 'grayscale_mask'],
                        help='visual prompt that will be marked on image')
    parser.add_argument('--expand_ratio', type=float, default=0.01,
                        help='ratio to expand around the boxes')
    parser.add_argument('--recompute_box', action='store_true', default=False,
                        help='If true, the box will be recomputed according to mask.')
    parser.add_argument('--color_line', type=str, default='red',
                        help='color of the box, circle, and contour')
    parser.add_argument('--color_mask', type=str, default='green',
                        help='color of the mask')
    parser.add_argument('--thickness', type=int, default=2,
                        help='thickness of the box, circle, and contour')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha of the mask')
    parser.add_argument("--blur_std_dev", type=int, default=100,
                        help="standard deviation of Gaussian blur")
    parser.add_argument("--contour_scale", type=float, default=1.0,
                        help="resize ratio for the contour mask")

    # clip args
    parser.add_argument('--clip_model', metavar='NAME', default='ViT-L/14@336px',
                        help='model architecture, e.g., ViT-L/14@336px, EVA02-CLIP-L-14-336')
    parser.add_argument('--clip_pretrained', metavar='NAME', default='',
                        help='path to model checkpoints')
    parser.add_argument('--clip_image_size', type=int, default=336,
                        help='input image size of clip')
    parser.add_argument('--clip_processing', default='resize',
                        choices=['center_crop', 'padding', 'bias_crop', 'resize'],
                        help='preprocessing operation to the clip input images one of (center_crop, padding)')
    parser.add_argument('--clip_crop_pct', type=float, default=1.0,
                        help='Only use it when clip_processing is center_crop.')
    # sam args
    parser.add_argument('--sam_prompt', default='grid', choices=['keypoint', 'box', 'grid'],
                        help='prompt that sam used to produce masks')
    parser.add_argument('--sam_model', metavar='NAME', default='vit_h',
                        help='model architecture (default: vit_h)')
    parser.add_argument('--sam_pretrained', metavar='NAME', default='',
                        help='path to model checkpoints')
    parser.add_argument('--sam_image_size', type=int, default=1024,
                        help='input image size of sam')
    parser.add_argument('--min_mask_region_area', default=400, type=int,
                        help='If >0, postprocessing will be applied to remove disconnected regions and holes in masks.')
    parser.add_argument('--sam_multimask_output', action='store_true', default=False,
                        help='If true, the model will return three masks.')
    parser.add_argument('--sam_neg_label', action='store_true', default=False,
                        help='If true, the model will also take negative points as inputs.')
    parser.add_argument('--points_per_side', type=int, default=16,
                        help='The number of points to be sampled along one side of the image.')
    parser.add_argument('--points_per_batch', type=int, default=256,
                        help='Sets the number of points run simultaneously by the model.')
    parser.add_argument('--pred_iou_thresh', type=float, default=0.86,
                        help="A filtering threshold in [0,1], using the model's predicted mask quality.")
    parser.add_argument('--stability_score_thresh', type=float, default=0.92,
                        help="A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.")
    parser.add_argument('--stability_score_offset', type=float, default=0.7,
                        help="The amount to shift the cutoff when calculated the stability score.")
    parser.add_argument('--box_nms_thresh', type=float, default=0.7,
                        help="The box IoU cutoff used by non-maximal suppression to filter duplicate masks.")
    parser.add_argument('--crop_n_layers', type=int, default=0,
                        help="If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.")
    parser.add_argument('--crop_nms_thresh', type=float, default=0.7,
                        help="The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.")
    parser.add_argument('--crop_overlap_ratio', type=float, default=512 / 1500,
                        help="Sets the degree to which crops overlap.")
    parser.add_argument('--crop_n_points_downscale_factor', type=int, default=2,
                        help="The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.")
    parser.add_argument('--point_grids', default=None,
                        help="A list over explicit grids of points used for sampling, normalized to [0,1].")
    parser.add_argument('--output_mode', type=str, default='binary_mask',
                        help="The form masks are returned in. Can be 'binary_mask', 'uncompressed_rle', or 'coco_rle'.")
    parser.add_argument('--filter_mask_thr', type=float, default=0.0,
                        help="If >0, filter the masks that are outside of the object.")

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print("loading CLIP")
    resize_transform_clip = ResizeLongestSide(args.clip_image_size)
    encoder, _ = clip.load(args.clip_model, download_root=args.clip_pretrained, device=device)
    tokenizer = clip.tokenize
    clip_model = ClipModel(encoder, tokenizer, device)

    print("loading SAM")
    sam_image_size = args.sam_image_size
    resize_transform_sam = ResizeLongestSide(sam_image_size)
    sam_model = sam_model_registry[args.sam_model](checkpoint=args.sam_pretrained).to(device)
    sam_mask_generator = SamAutomaticMaskGenerator(
        sam_model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=args.stability_score_offset,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        crop_overlap_ratio=args.crop_overlap_ratio,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        point_grids=args.point_grids,
        min_mask_region_area=args.min_mask_region_area,
        output_mode=args.output_mode,
    )

    print("loading PROMPT")
    pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1).to(device) * 255.0
    pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1).to(device) * 255.0
    fgvp = FGVP_ENSEMBLE(
        color_line=args.color_line,
        thickness=args.thickness,
        color_mask=args.color_mask,
        alpha=args.alpha,
        clip_processing=args.clip_processing,
        clip_image_size=args.clip_image_size,
        resize_transform_clip=resize_transform_clip,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        blur_std_dev=args.blur_std_dev,
        mask_threshold=sam_model.mask_threshold,
        contour_scale=args.contour_scale,
        device=device,
    )

    # load image
    image = cv2.imread(args.img_dir)
    real_size = image.shape[:2]

    boxes, masks = get_masks(args, image)
    centers = torch.stack([boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)], 1)

    res = image.copy()
    for box, mask in zip(boxes, masks):
        box = box.detach().cpu().numpy().astype(int)
        mask = mask.squeeze().detach().cpu().numpy()
        res = draw_box(res, box, (0, 255, 0))
        res = draw_mask(res, mask, (0, 255, 0))
    cv2.imwrite(os.path.join(args.out_dir, "candidates.jpg"), res)

    text_inputs = [f"a photo of a {t}" for t in args.text]
    clip_inputs = torch.cat([fgvp(vp, image[:, :, ::-1], centers, boxes, masks) for vp in args.visual_prompt])

    # os.makedirs(os.path.join(args.out_dir, 'fgvp'), exist_ok=True)
    # for idx, clip_input in enumerate(clip_inputs):
    #     clip_input = clip_input * pixel_std + pixel_mean
    #     clip_input = clip_input.permute(1, 2, 0).cpu().numpy().astype(np.uint8)[..., ::-1]
    #     cv2.imwrite(os.path.join(args.out_dir, f"fgvp/{idx}.jpg"), clip_input)

    logits_per_image = clip_model(clip_inputs, text_inputs)
    logits_per_image = logits_per_image.view(len(args.visual_prompt), len(boxes), len(text_inputs)).mean(0)

    # scores, row_inds = logits_per_image.topk(1, dim=1)
    N, M = logits_per_image.shape
    cost = torch.exp(-logits_per_image)
    row_inds, col_inds = linear_sum_assignment(cost.cpu().numpy())

    res = image.copy()
    print(boxes[row_inds])
    for row_ind, col_ind in zip(row_inds, col_inds):
        box = boxes[row_ind].detach().cpu().numpy().astype(int)
        mask = masks[row_ind].squeeze().detach().cpu().numpy()
        caption = args.text[col_ind]
        res = draw_box(res, box, (0, 255, 0))
        res = draw_mask(res, mask, (0, 255, 0))
        res = draw_text(res, caption, box[0], box[1], (0, 255, 0))
    cv2.imwrite(os.path.join(args.out_dir, "res.jpg"), res)
