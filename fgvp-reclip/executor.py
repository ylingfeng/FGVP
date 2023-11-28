from typing import List, Dict, Union, Tuple

from PIL import Image, ImageDraw, ImageFilter
import spacy
import hashlib
import os

import torch
import torchvision
import torchvision.transforms as transforms
import clip
from transformers import BertTokenizer, RobertaTokenizerFast
import ruamel.yaml as yaml

from interpreter import Box
from albef.model import ALBEF
from albef.utils import *
from albef.vit import interpolate_pos_embed

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import batched_mask_to_box, remove_small_regions
import clip
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transforms import ResizeShortestSide, bias_crop, boxes_to_circles, str2rgb
from show import show_box, show_masks, show_points
from fine_grained_visual_prompt import FGVP_ENSEMBLE


class Executor:
    def __init__(self, args=None, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None) -> None:
        IMPLEMENTED_METHODS = ["crop", "blur", "shade", "fine-grained"]
        if any(m not in IMPLEMENTED_METHODS for m in box_representation_method.split(",")):
            raise NotImplementedError
        IMPLEMENTED_AGGREGATORS = ["max", "sum"]
        if method_aggregator not in IMPLEMENTED_AGGREGATORS:
            raise NotImplementedError
        self.box_representation_method = box_representation_method
        self.split = []
        for method in self.box_representation_method.split(','):
            if method == "fine-grained":
                self.split.append(len(args.visual_prompt))
            else:
                self.split.append(1)
        self.method_aggregator = method_aggregator
        self.enlarge_boxes = enlarge_boxes
        self.device = device
        self.expand_position_embedding = expand_position_embedding
        self.square_size = square_size
        self.blur_std_dev = blur_std_dev
        self.cache_path = cache_path
        self.args = args
        self.recompute_box = args.recompute_box if args is not None else False
        # self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        # self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)
        self.pixel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(device) * 255.0
        self.pixel_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(device) * 255.0

        if "fine-grained" in self.box_representation_method:
            self.sam_model = sam_model_registry[args.sam_model](checkpoint=args.sam_pretrained).to(self.device)
            # clip_model, _ = clip.load(args.clip_model, download_root=args.clip_pretrained, device=self.device)
            # clip_tokenizer = clip.tokenize

            # sam transform
            self.sam_image_size = args.sam_image_size
            self.resize_transform_sam = ResizeLongestSide(self.sam_image_size)
            self.sam_mask_generator = SamAutomaticMaskGenerator(
                self.sam_model,
                points_per_side=16,
                points_per_batch=256,
                pred_iou_thresh=0.86,  # 0.86,
                stability_score_thresh=0.92,  # 0.92,
                stability_score_offset=0.7,
                box_nms_thresh=0.7,
                crop_n_layers=0,
                crop_nms_thresh=0.7,
                crop_overlap_ratio=512 / 1500,
                crop_n_points_downscale_factor=2,
                point_grids=None,
                min_mask_region_area=0,
                output_mode="binary_mask",
            )
            # clip transform
            self.clip_image_size = args.clip_image_size
            if args.clip_processing == 'padding':
                self.resize_transform_clip = ResizeLongestSide(self.clip_image_size)
            else:
                self.resize_transform_clip = ResizeShortestSide(self.clip_image_size / args.clip_crop_pct)
            self.fgvp = FGVP_ENSEMBLE(
                color_line=args.color_line,
                thickness=args.thickness,
                color_mask=args.color_mask,
                alpha=args.alpha,
                clip_processing=args.clip_processing,
                clip_image_size=self.clip_image_size,
                resize_transform_clip=self.resize_transform_clip,
                pixel_mean=self.pixel_mean,
                pixel_std=self.pixel_std,
                blur_std_dev=blur_std_dev,
                mask_threshold=self.sam_model.mask_threshold,
                contour_scale=args.contour_scale,
                device=device,
            )

    def preprocess_image(self, image: Image) -> List[torch.Tensor]:
        return [preprocess(image) for preprocess in self.preprocesses]

    def preprocess_text(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def tensorize_inputs(self, caption: str, image: Image, boxes: List[Box], image_name: str = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        returned_boxes = None
        args = self.args
        images = []
        for preprocess in self.preprocesses:
            images.append([])
        if self.cache_path is None or any([not os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")) for model_name in self.model_names for method_name in self.box_representation_method.split(',')]):
            returned_boxes = boxes[:]
            if "crop" in self.box_representation_method:
                for i in range(len(boxes)):
                    image_i = image.copy()
                    box = [
                        max(boxes[i].left-self.enlarge_boxes, 0),
                        max(boxes[i].top-self.enlarge_boxes, 0),
                        min(boxes[i].right+self.enlarge_boxes, image_i.width),
                        min(boxes[i].bottom+self.enlarge_boxes, image_i.height)
                    ]
                    image_i = image_i.crop(box)
                    preprocessed_images = self.preprocess_image(image_i)
                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))
            if "blur" in self.box_representation_method:
                for i in range(len(boxes)):
                    image_i = image.copy()
                    mask = Image.new('L', image_i.size, 0)
                    draw = ImageDraw.Draw(mask)
                    box = (
                        max(boxes[i].left-self.enlarge_boxes, 0),
                        max(boxes[i].top-self.enlarge_boxes, 0),
                        min(boxes[i].right+self.enlarge_boxes, image_i.width),
                        min(boxes[i].bottom+self.enlarge_boxes, image_i.height)
                    )
                    draw.rectangle([box[:2], box[2:]], fill=255)
                    blurred = image_i.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    blurred.paste(image_i, mask=mask)
                    preprocessed_images = self.preprocess_image(blurred)
                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))
            if "shade" in self.box_representation_method:
                for i in range(len(boxes)):
                    TINT_COLOR = (240, 0, 30)
                    image_i = image.copy().convert('RGBA')
                    overlay = Image.new('RGBA', image_i.size, TINT_COLOR+(0,))
                    draw = ImageDraw.Draw(overlay)
                    box = [
                        max(boxes[i].left-self.enlarge_boxes, 0),
                        max(boxes[i].top-self.enlarge_boxes, 0),
                        min(boxes[i].right+self.enlarge_boxes, image_i.width),
                        min(boxes[i].bottom+self.enlarge_boxes, image_i.height)
                    ]
                    draw.rectangle((tuple(box[:2]), tuple(box[2:])), fill=TINT_COLOR+(127,))
                    shaded_image = Image.alpha_composite(image_i, overlay)
                    shaded_image = shaded_image.convert('RGB')
                    preprocessed_images = self.preprocess_image(shaded_image) # []
                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))
            if "fine-grained" in self.box_representation_method:
                image = np.array(image)
                new_size = real_size = image.shape[:2]
                boxes = [[b.left, b.top, b.right, b.bottom] for b in boxes]
                boxes = torch.Tensor(boxes)
                centers = torch.stack([boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)], 1)

                if 'none' in args.sam_prompt:
                    masks = torch.zeros(len(boxes), 1, *image.shape[:2])
                elif "grid" in args.sam_prompt:
                    # load sam prompting from grids
                    ori_size = image.shape[:2]
                    image = self.resize_transform_sam.apply_image(image)  # np.array(rgb)
                    new_size = image.shape[:2]
                    outputs = self.sam_mask_generator.generate(image)
                    masks = torch.from_numpy(np.stack([x['segmentation'] for x in outputs])).unsqueeze(1)
                    boxes = torch.from_numpy(np.stack([x['bbox'] for x in outputs])).float()
                    centers = torch.from_numpy(np.concatenate([x['point_coords'] for x in outputs])).float()
                    boxes[:, 2] += boxes[:, 0]
                    boxes[:, 3] += boxes[:, 1]
                    # centers = torch.stack([boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)], 1)
                    # end sam
                else:
                    assert 'box' in args.sam_prompt
                    # load sam
                    ori_size = image.shape[:2]
                    image = self.resize_transform_sam.apply_image(image)
                    new_size = image.shape[:2]
                    boxes = self.resize_transform_sam.apply_boxes_torch(boxes, ori_size)

                    sam_inputs = torch.as_tensor(image, device=self.device)
                    sam_inputs = sam_inputs.permute(2, 0, 1).contiguous()[None, :, :, :]
                    sam_inputs = self.sam_model.preprocess(sam_inputs)

                    in_boxes = boxes[:, None, :].to(self.device)

                    features = self.sam_model.image_encoder(sam_inputs)
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(points=None, boxes=in_boxes, masks=None)
                    low_res_masks, iou_pred = self.sam_model.mask_decoder(
                        image_embeddings=features,
                        image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=args.sam_multimask_output,
                    )
                    masks = F.interpolate(low_res_masks, (self.sam_image_size, self.sam_image_size),
                                        mode="bilinear", align_corners=False)  # N, 1, H, W
                    masks = masks[:, -1:, :new_size[0], :new_size[1]]

                    if args.min_mask_region_area > 0:
                        bit_masks = masks > self.sam_model.mask_threshold
                        masks = []
                        for mask in bit_masks:
                            mask = mask.squeeze(0).cpu().numpy()
                            mask, changed = remove_small_regions(mask, args.min_mask_region_area, mode="holes")
                            mask, changed = remove_small_regions(mask, args.min_mask_region_area, mode="islands")
                            mask = torch.as_tensor(mask, device=self.device).unsqueeze(0)
                            masks.append(mask)
                        masks = torch.stack(masks, 0)
                    if args.recompute_box:
                        bit_masks = masks > self.sam_model.mask_threshold
                        boxes = batched_mask_to_box(bit_masks.squeeze(1)).float()
                        centers = torch.stack([boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)], 1)
                clip_inputs = torch.cat([self.fgvp(vp, image, centers, boxes, masks) for vp in args.visual_prompt])
        
                ratio_h, ratio_w = real_size[0] / new_size[0], real_size[1] / new_size[1]
                boxes[:, 0::2] *= ratio_w
                boxes[:, 1::2] *= ratio_h
                returned_boxes = [Box(x=b[0].item(), y=b[1].item(), w=(b[2]-b[0]).item(), h=(b[3]-b[1]).item()) for b in boxes]

                for j, preprocess in enumerate(self.preprocesses):
                    size = preprocess.transforms[0].size
                    res = F.interpolate(clip_inputs.clone(), size, mode='bilinear', align_corners=False)
                    res = [r.squeeze(0) for r in res.split(1, 0)]
                    images[j] += res

            imgs = [torch.stack(image_list) for image_list in images]
        else:
            imgs = [[] for _ in self.models]

        return imgs, returned_boxes

    @torch.no_grad()
    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None, caption_bank=[]) -> torch.Tensor:

        images, returned_boxes = self.tensorize_inputs(caption, image, boxes, image_name)
        text_tensor = self.preprocess_text([caption] + caption_bank).to(self.device)

        all_logits_per_image = []
        all_logits_per_text = []
        box_representation_methods = self.box_representation_method.split(',')
        caption_hash = hashlib.md5(caption.encode('utf-8')).hexdigest()
        for model, images_t, model_name in zip(self.models, images, self.model_names):
            if self.cache_path is not None:
                text_cache_path = os.path.join(self.cache_path, model_name, "text"+("_shade" if self.box_representation_method == "shade" else ""))
            image_features = None
            text_features = None
            if self.cache_path is not None and os.path.exists(os.path.join(self.cache_path, model_name)):
                if os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")):
                    try:
                        text_features = torch.load(os.path.join(text_cache_path, caption_hash+".pt"), map_location=self.device)
                    except:
                        print(os.path.join(text_cache_path, caption_hash+".pt"))
                        exit()
                if os.path.exists(os.path.join(self.cache_path, model_name, image_name)):
                    if all([os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")) for method_name in box_representation_methods]):
                        image_features = []
                        for split, method_name in zip(self.split, box_representation_methods):
                            features = torch.load(os.path.join(self.cache_path, model_name, image_name, method_name+".pt"), map_location=self.device)
                            features = [features[(box.x, box.y, box.w, box.h)] for box in boxes]
                            features = torch.stack(features, dim=-2).reshape(split * len(boxes), -1)
                            image_features.append(features)
                        image_features = torch.cat(image_features)
                        # image_features = image_features.view(-1, image_features.shape[-1])
                    if self.recompute_box:
                        if os.path.exists(os.path.join(self.cache_path, model_name, image_name, "box.pt")):
                            assert returned_boxes is None
                            box_mapping = torch.load(os.path.join(self.cache_path, model_name, image_name, "box.pt"), map_location=self.device)
                            cached_boxes = [box_mapping[(box.x, box.y, box.w, box.h)] for box in boxes]
            logits_per_image, logits_per_text, image_features, text_features = self.call_model(model, images_t, text_tensor, image_features=image_features, text_features=text_features)
            all_logits_per_image.append(logits_per_image)
            all_logits_per_text.append(logits_per_text)
            if self.cache_path is not None and image_name is not None and image_features is not None:
                image_features = image_features.view(-1, len(boxes), image_features.shape[-1])
                image_features_list = image_features.split(self.split, dim=0)
                if not os.path.exists(os.path.join(self.cache_path, model_name, image_name)):
                    os.makedirs(os.path.join(self.cache_path, model_name, image_name))
                # for i in range(image_features.shape[0]):
                for split, features, method_name in zip(self.split, image_features_list, box_representation_methods):
                    if not os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")):
                        features = features.reshape(split, len(boxes), image_features.shape[-1])
                        image_features_dict = {(box.x, box.y, box.w, box.h): features[:, j, :].cpu() for j, box in enumerate(boxes)}
                        torch.save(image_features_dict, os.path.join(self.cache_path, model_name, image_name, method_name+".pt"))
                if self.recompute_box:
                    if not os.path.exists(os.path.join(self.cache_path, model_name, image_name, "box.pt")):
                        assert returned_boxes is not None
                        box_mapping_dict = {(box.x, box.y, box.w, box.h): re_box for box, re_box in zip(boxes, returned_boxes)}
                        torch.save(box_mapping_dict, os.path.join(self.cache_path, model_name, image_name, "box.pt"))
            if self.cache_path is not None and not os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")) and text_features is not None:
                assert text_features.shape[0] == 1 or self.args.score_subtracting
                if not os.path.exists(text_cache_path):
                    os.makedirs(text_cache_path)
                torch.save(text_features.cpu(), os.path.join(text_cache_path, caption_hash+".pt"))

        all_logits_per_image = torch.stack(all_logits_per_image).sum(0)
        all_logits_per_text = torch.stack(all_logits_per_text).sum(0)  # 1, n_boxes * n_prompt
        if self.args.score_subtracting:
            # n_captions, n_boxes * n_prompt
            pos_sample = all_logits_per_text[:1]
            neg_sample = all_logits_per_text[1:].mean(0, keepdim=True)
            all_logits_per_text = pos_sample - neg_sample

        # 1 * n_prompt, n_boxes
        if self.method_aggregator == "max":
            all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).max(dim=0, keepdim=True)[0]
        elif self.method_aggregator == "sum":
            all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).sum(dim=0, keepdim=True)
        if self.recompute_box:
            boxes = cached_boxes if returned_boxes is None else returned_boxes
        return all_logits_per_text.view(-1), boxes

class ClipExecutor(Executor):
    def __init__(self, args=None, clip_model: str = "ViT-B/32", device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None) -> None:
        super().__init__(args, device, box_representation_method, method_aggregator, enlarge_boxes, expand_position_embedding, square_size, blur_std_dev, cache_path)
        self.clip_models = clip_model.split(",")
        self.model_names = [model_name.replace("/", "_") for model_name in self.clip_models]
        self.models = []
        self.preprocesses = []
        for model_name in self.clip_models:
            model, preprocess = clip.load(model_name, device=device, jit=False)
            self.models.append(model)
            if self.square_size:
                print("Square size!")
                preprocess.transforms[0] = transforms.Resize((model.visual.input_resolution, model.visual.input_resolution), interpolation=transforms.InterpolationMode.BICUBIC)
            self.preprocesses.append(preprocess)
        self.models = torch.nn.ModuleList(self.models)

    def preprocess_text(self, text: List[str]) -> torch.Tensor:
        return clip.tokenize(text)

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: torch.Tensor, image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
        if image_features is None:
            print('computing image features')
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if text_features is None:
            print('computing text features')
            text_features = model.encode_text(text)
            # normalized features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, image_features, text_features

    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None, caption_bank=[]) -> torch.Tensor:
        if "shade" in self.box_representation_method:
            caption = caption.lower() + " is in red color."
        elif self.args.text_prompt == 'a photo of <caption>':
            caption = f"a photo of {caption.lower()}"
        elif self.args.text_prompt == 'This is <caption>':
            caption = f"This is {caption.lower()}"
        else:
            raise NotImplementedError

        if self.expand_position_embedding:
            original_preprocesses = self.preprocesses
            new_preprocesses = []
            original_position_embeddings = []
            for model_name, model, preprocess in zip(self.clip_models, self.models, self.preprocesses):
                if "RN" in model_name:
                    model_spatial_dim = int((model.visual.attnpool.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.attnpool.positional_embedding.clone()
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.attnpool.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.attnpool.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                else:
                    model_spatial_dim = int((model.visual.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.positional_embedding.clone()
                    model.visual.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                new_preprocesses.append(transform)
                original_position_embeddings.append(original_positional_embedding)
            self.preprocesses = new_preprocesses
        result, boxes = super().__call__(caption, image, boxes, image_name, caption_bank)
        if self.expand_position_embedding:
            self.preprocesses = original_preprocesses
            for model, model_name, pos_embedding in zip(self.models, self.clip_models, original_position_embeddings):
                if "RN" in model_name:
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(pos_embedding)
                else:
                    model.visual.positional_embedding = torch.nn.Parameter(pos_embedding)
        return result, boxes

class ClipGradcamExecutor(ClipExecutor):
    def __init__(self, args=None, clip_model: str = "ViT-B/32", device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", gradcam_alpha: List[float] = [1.0], expand_position_embedding: bool = False, background_subtract: bool = False, square_size: bool = False, blur_std_dev: int = 100, gradcam_ensemble_before: bool = False) -> None:
        super().__init__(args, clip_model, device, box_representation_method, method_aggregator, False, expand_position_embedding, square_size, blur_std_dev, None)
        self.clip_models = clip_model.split(",")
        for i in range(len(self.clip_models)):
            if "ViT" in self.clip_models[i]:
                import clip_mm_explain
                self.models[i] = clip_mm_explain.load(self.clip_models[i], device=device, jit=False)[0]
        self.gradcam_alpha = gradcam_alpha
        self.expand_position_embedding = expand_position_embedding
        self.background_subtract = background_subtract
        self.gradcam_ensemble_before = gradcam_ensemble_before

    def __call__(self, caption: str, image: Image, boxes: List[Box], return_gradcam=False, image_name: str = None, caption_bank=[]) -> torch.Tensor:
        if self.background_subtract:
            self.background_subtract = False
            background = self("", image, boxes, True)
            self.background_subtract = True
        text_tensor = self.preprocess_text(caption).to(self.device)
        scores_list = []
        gradcam_list = []
        for model_name, model, preprocess, gradcam_alpha in zip(self.clip_models, self.models, self.preprocesses, self.gradcam_alpha):
            if "RN" in model_name:
                if self.expand_position_embedding:
                    model_spatial_dim = int((model.visual.attnpool.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.attnpool.positional_embedding.clone()
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.attnpool.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.attnpool.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                    image_t = transform(image).unsqueeze(0).to(self.device)
                    print(model.visual.attnpool.positional_embedding.shape, image_t.shape, model_spatial_dim, patch_size, image.size)
                else:
                    image_t = preprocess(image).unsqueeze(0).to(self.device)
                activations_and_grads = ActivationsAndGradients(model, [model.visual.layer4], None)
                height_width_ratio = image_t.shape[2] / image_t.shape[1]
                image_t = torch.autograd.Variable(image_t)
                logits_per_image, logits_per_text = activations_and_grads(image_t, text_tensor)
                logits = torch.diagonal(logits_per_image, 0)
                loss = logits.sum()
                loss.backward()
                grads = activations_and_grads.gradients[0].mean(dim=(2, 3), keepdim=True)
                gradcam = (grads*activations_and_grads.activations[0]).sum(1, keepdim=True).float().clamp(min=0)
                assert len(gradcam.shape) == 4
                gradcam = torch.nn.functional.interpolate(gradcam,size = (image.height,image.width), mode='bicubic').squeeze()
                if self.expand_position_embedding:
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(original_positional_embedding)
            else:
                model_spatial_dim = int((model.visual.positional_embedding.shape[0]-1)**0.5)
                patch_size = model.visual.input_resolution // model_spatial_dim
                if self.expand_position_embedding:
                    original_positional_embedding = model.visual.positional_embedding.clone()
                    model.visual.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                    image_t = transform(image).unsqueeze(0).to(self.device)
                else:
                    image_t = preprocess(image).unsqueeze(0).to(self.device)
                logits_per_image, logits_per_text = model(image_t, text_tensor)
                loss = logits_per_image.sum()
                model.zero_grad()
                loss.backward()
                image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
                num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
                R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(self.device)
                for block in image_attn_blocks[-1:]:
                    grad = block.attn_grad
                    cam = block.attn_probs
                    print(cam.shape, grad.shape, num_tokens, image_t.shape, patch_size, model_spatial_dim)
                    cam = cam.view(-1, cam.shape[-1], cam.shape[-1])
                    grad = grad.view(-1, grad.shape[-1], grad.shape[-1])
                    cam = grad * cam
                    cam = cam.clamp(min=0).mean(dim=0)
                    R += torch.matmul(cam, R)
                if self.expand_position_embedding:
                    gradcam = R[0,1:].view(1, 1, image.height // patch_size, image.width // patch_size)
                    model.visual.positional_embedding = torch.nn.Parameter(original_positional_embedding)
                else:
                    gradcam = R[0,1:].view(1, 1, model_spatial_dim, model_spatial_dim)
                gradcam = torch.nn.functional.interpolate(gradcam, size=(image.height, image.width), mode='bicubic', align_corners=False).view(image.height, image.width)
            if self.background_subtract:
                gradcam = gradcam - background
            if return_gradcam:
                return gradcam, boxes
            scores = []
            for box in boxes:
                det_area = box.area
                score = gradcam[int(box.top):int(box.bottom),int(box.left):int(box.right)]
                score = score.sum() / det_area**gradcam_alpha
                scores.append(score)
            scores_list.append(torch.stack(scores).detach())
            gradcam_list.append(gradcam)
        scores = torch.stack(scores_list).mean(0)
        if self.gradcam_ensemble_before:
            gradcam = torch.stack(gradcam_list).mean(0)
            scores = []
            for box in boxes:
                det_area = box.area
                score = gradcam[int(box.top):int(box.bottom),int(box.left):int(box.right)]
                score = score.sum() / det_area**gradcam_alpha
                scores.append(score)
            scores = torch.stack(scores).detach()
        return scores, boxes

class AlbefExecutor(Executor):
    def __init__(self, args, checkpoint_path: str, config_path: str, max_words: int = 30, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", mode: str = "itm", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None) -> None:
        super().__init__(args, device, box_representation_method, method_aggregator, enlarge_boxes, expand_position_embedding, square_size, blur_std_dev, cache_path)
        if device == "cpu":
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            checkpoint = torch.load(checkpoint_path)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        self.image_res = config["image_res"]
        bert_model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model_names = ["albef_"+mode]


        model = ALBEF(config=config, text_encoder=bert_model_name, tokenizer=self.tokenizer)
        model = model.to(self.device)

        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        if 'visual_encoder_m.pos_embed':
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

        model.eval()
        model.logit_scale = 1./model.temp
        self.models = torch.nn.ModuleList(
            [
                model
            ]
        )
        self.image_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]
        )
        self.preprocesses = [self.image_transform]
        self.max_words = max_words
        self.mode = mode

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        if "shade" in self.box_representation_method:
            modified_text = pre_caption(text+" is in red color.", self.max_words)
        else:
            modified_text = pre_caption(text, self.max_words)
        text_input = self.tokenizer(modified_text, padding='longest', return_tensors="pt")
        sep_mask = text_input.input_ids == self.tokenizer.sep_token_id
        text_input.attention_mask[sep_mask] = 0
        return text_input

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: Dict[str, torch.Tensor], image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
        image_feat = image_features
        text_feat = text_features
        if self.mode == "itm":
            image_embeds = model.visual_encoder(images)
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
            output = model.text_encoder(
                text.input_ids,
                attention_mask = text.attention_mask,
                encoder_hidden_states = image_embeds,
                encoder_attention_mask = image_atts,
                return_dict = True,
            )
            vl_embeddings = output.last_hidden_state[:,0,:]
            vl_output = model.itm_head(vl_embeddings)
            logits_per_image = vl_output[:,1:2]
            logits_per_text = logits_per_image.permute(1, 0)
            image_feat = None
            text_feat = None
        else:
            if image_feat is None:
                image_embeds = model.visual_encoder(images, register_blk=-1)
                image_feat = torch.nn.functional.normalize(model.vision_proj(image_embeds[:,0,:]),dim=-1)
            if text_feat is None:
                text_output = model.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                                 return_dict = True, mode = 'text')
                text_embeds = text_output.last_hidden_state
                text_feat = torch.nn.functional.normalize(model.text_proj(text_embeds[:,0,:]),dim=-1)
            sim = image_feat@text_feat.t()/model.temp
            logits_per_image = sim
            logits_per_text = sim.t()
        return logits_per_image, logits_per_text, image_feat, text_feat

class AlbefGradcamExecutor(AlbefExecutor):
    def __init__(self, args, checkpoint_path: str, config_path: str, max_words: int = 30, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", gradcam_alpha: float = 1.0, gradcam_mode: str = "itm", block_num: int = 8, enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False) -> None:
        super().__init__(args, checkpoint_path, config_path, max_words, device, box_representation_method, method_aggregator, gradcam_mode, enlarge_boxes, expand_position_embedding, square_size, None, None)
        self.gradcam_alpha = gradcam_alpha
        self.gradcam_mode = gradcam_mode
        self.block_num = block_num
        self.model = self.models[0]

    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None, caption_bank=[]) -> torch.Tensor:
        self.model.text_encoder.base_model.base_model.encoder.layer[self.block_num].crossattention.self.save_attention = True
        text_input = self.preprocess_text(caption).to(self.device)
        image_t = self.preprocesses[0](image).unsqueeze(0).to(self.device)

        if self.gradcam_mode=='itm':
            full_gradcam = []
            for txt_input in [text_input]:
                image_embeds = self.model.visual_encoder(image_t)
                image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image_t.device)
                output = self.model.text_encoder(txt_input.input_ids,
                                        attention_mask = txt_input.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        return_dict = True,
                                       )

                vl_embeddings = output.last_hidden_state[:,0,:]
                vl_output = self.model.itm_head(vl_embeddings)
                loss = vl_output[:,1].sum()

                self.model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    mask = txt_input.attention_mask.view(txt_input.attention_mask.size(0),1,-1,1,1)

                    grads = self.model.text_encoder.base_model.base_model.encoder.layer[self.block_num].crossattention.self.get_attn_gradients().detach()
                    cams = self.model.text_encoder.base_model.base_model.encoder.layer[self.block_num].crossattention.self.get_attention_map().detach()

                    cams = cams[:, :, :, 1:].reshape(image_t.size(0), 12, -1, 24, 24) * mask
                    grads = grads[:, :, :, 1:].clamp(min=0).reshape(image_t.size(0), 12, -1, 24, 24) * mask

                    gradcam = cams * grads
                    gradcam = gradcam.mean(1).mean(1)
                full_gradcam.append(gradcam)
        if self.gradcam_mode=='itc':
            image_embeds = self.model.visual_encoder(image_t, register_blk=self.block_num)
            image_feat = torch.nn.functional.normalize(self.model.vision_proj(image_embeds[:,0,:]),dim=-1)
            text_output = self.model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask,
                                             return_dict = True, mode = 'text')
            text_embeds = text_output.last_hidden_state
            text_feat = torch.nn.functional.normalize(self.model.text_proj(text_embeds[:,0,:]),dim=-1)
            sim = image_feat@text_feat.t()/self.model.temp
            loss = sim.diag().sum()

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad = self.model.visual_encoder.blocks[self.block_num].attn.get_attn_gradients().detach()
                cam = self.model.visual_encoder.blocks[self.block_num].attn.get_attention_map().detach()
                cam = cam[:, :, 0, 1:].reshape(image_t.size(0), -1, 24, 24)
                grad = grad[:, :, 0, 1:].reshape(image_t.size(0), -1, 24, 24).clamp(0)
                gradcam = (cam * grad).mean(1)
            full_gradcam = [gradcam]
        gradcam = torch.stack(full_gradcam).sum(0)
        gradcam = gradcam.view(1,1,int(gradcam.numel()**0.5), int(gradcam.numel()**0.5))
        gradcam = torch.nn.functional.interpolate(gradcam,size = (image.height,image.width), mode='bicubic').squeeze()
        scores = []
        for box in boxes:
            det_area = box.area
            score = gradcam[int(box.top):int(box.bottom),int(box.left):int(box.right)]
            score = score.sum() / det_area**self.gradcam_alpha
            scores.append(score)
        return torch.stack(scores).to(self.device), boxes

class MdetrExecutor(Executor):
    def __init__(self, args, model_name: str, device: str = "cpu", use_token_mapping: bool = False, freeform_bboxes: bool = True, enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100):
        super().__init__(args, device, "crop", "max", enlarge_boxes, expand_position_embedding, square_size, blur_std_dev)
        self.model, self.postprocessor = torch.hub.load('ashkamath/mdetr:main', model_name, pretrained=True, return_postprocessor=True)
        self.model = self.model.to(device)
        self.model.eval()
        # standard PyTorch mean-std input image normalization
        self.transform = transforms.Compose([
            transforms.Resize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.box_recall = [0, 0]
        self.use_token_mapping = use_token_mapping
        if self.use_token_mapping:
            self.nlp = spacy.load("en_core_web_sm")
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.freeform_bboxes = freeform_bboxes

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(self.device)
        return b

    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None, caption_bank=[]) -> torch.Tensor:
        with torch.no_grad():
            image_t = self.transform(image).unsqueeze(0).to(self.device)
            memory_cache = self.model(image_t, [caption], encode_and_save=True)
            outputs = self.model(image_t, [caption], encode_and_save=False, memory_cache=memory_cache)
        if self.use_token_mapping:
            doc = self.nlp(caption)
            head_index = -1
            for i in range(len(doc)):
                if doc[i].head.i == i:
                    head_index = i
                    break
            tokens_info = self.tokenizer.encode_plus(caption, return_offsets_mapping=True)
            wp_head_indices = [i for i in range(len(tokens_info["offset_mapping"][1:])) if tokens_info["offset_mapping"][i][0] >= doc[head_index].idx and tokens_info["offset_mapping"][i][0] < doc[head_index].idx+len(doc[head_index].text)]
            probabilities = outputs['pred_logits'].softmax(-1)[0,:,wp_head_indices].sum(-1).to(self.device)
        else:
            probabilities = 1 - outputs['pred_logits'].softmax(-1)[0,:,-1].to(self.device)
        if freeform_bboxes:
            keep = [probabilities.argmax().item()]
            bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'].to(self.device)[0,keep,:], image.size)
            logits = (probabilities[keep]+1e-8).log()
            return logits, bboxes_scaled
        keep = list(range(outputs['pred_boxes'].shape[1]))
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'].to(self.device)[0,keep,:], image.size)
        given_boxes_tensor = torch.FloatTensor([[box.left, box.top, box.right, box.bottom] for box in boxes]).to(self.device)
        ious = torchvision.ops.boxes.box_iou(given_boxes_tensor, bboxes_scaled)
        box_indices = [ious[i,:].argmax().item() for i in range(len(boxes))]
        for i in range(len(boxes)):
            if ious[i,box_indices[i]].item() >= 0.8:
                self.box_recall[0] += 1
            self.box_recall[1] += 1
        return (probabilities[box_indices]+1e-8).log(), boxes

    def get_box_recall(self):
        return self.box_recall[0]/self.box_recall[1]
