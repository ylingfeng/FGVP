from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

from transforms import bias_crop, boxes_to_circles, scale_contour, scale_mask, str2rgb


class FGVP_ENSEMBLE:
    def __init__(
        self,
        color_line,
        thickness,
        color_mask,
        alpha,
        clip_processing,
        clip_image_size,
        resize_transform_clip,
        pixel_mean,
        pixel_std,
        blur_std_dev,
        mask_threshold=0.0,
        contour_scale=1.0,
        device='cpu',
    ):
        self.color_line = color_line
        self.thickness = thickness
        self.color_mask = color_mask
        self.alpha = alpha
        self.clip_processing = clip_processing
        self.clip_image_size = clip_image_size
        self.resize_transform_clip = resize_transform_clip
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.blur_std_dev = blur_std_dev
        self.mask_threshold = mask_threshold
        self.contour_scale = contour_scale
        self.device = device

    def __call__(self, visual_prompt: str, image, centers, boxes, masks):
        assert len(centers) == len(boxes) == len(masks)
        if 'crop' == visual_prompt or 'cpt' == visual_prompt or 'cpt_seg' == visual_prompt or 'ov_seg' == visual_prompt or 'blur_seg' == visual_prompt:
            clip_inputs = []
            for box, mask in zip(boxes, masks):
                box = box.int().cpu().numpy()
                if box[1] == box[3]:
                    box[3] += 1
                if box[0] == box[2]:
                    box[2] += 1
                res = image.copy()[box[1]:box[3], box[0]:box[2]]
                mask = mask[:, box[1]:box[3], box[0]:box[2]].unsqueeze(0)
                ori_size = res.shape[:2]
                res = self.resize_transform_clip.apply_image(res)
                new_size = res.shape[:2]
                mask = F.interpolate(mask.float(), (new_size[0], new_size[1]),
                                     mode="bilinear", align_corners=False)  # N, 1, H, W
                mask = (mask > self.mask_threshold).squeeze(1).squeeze(0).cpu().numpy()
                if 'cpt' == visual_prompt:
                    overlay = np.array(str2rgb(self.color_mask))[None, None, :].repeat(
                        res.shape[0], 0).repeat(res.shape[1], 1).astype(np.uint8)
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                elif 'cpt_seg' == visual_prompt:
                    overlay = res.copy()
                    overlay[mask == 1] = np.array(str2rgb(self.color_mask))
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                elif 'ov_seg' == visual_prompt:
                    res[mask == 0] = np.array(str2rgb(self.color_mask))
                elif 'blur_seg' == visual_prompt:
                    res = Image.fromarray(res)
                    overlay = res.copy()
                    overlay = overlay.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    overlay.paste(res, mask=Image.fromarray(mask))
                    res = np.array(overlay)
                else:
                    assert 'crop' == visual_prompt
                res = torch.from_numpy(res).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
                # rgb -> normalized
                res = (res - self.pixel_mean) / self.pixel_std
                if self.clip_processing == 'resize':
                    res = F.interpolate(res, (self.clip_image_size, self.clip_image_size),
                                        mode='bilinear', align_corners=False)
                else:
                    # Pad or left top crop
                    h, w = res.shape[-2:]
                    padh = self.clip_image_size - h
                    padw = self.clip_image_size - w
                    res = F.pad(res, (0, padw, 0, padh))  # if resize shortest side, this means left top crop
                    # res = F.pad(res, (math.floor(padw / 2), math.ceil(padw / 2),
                    #             math.floor(padh / 2), math.ceil(padh / 2)))  # if resize shortest side, this means center crop
                clip_inputs.append(res)
            clip_inputs = torch.cat(clip_inputs)
        else:
            # laod clip
            ori_size = image.shape[:2]
            image = self.resize_transform_clip.apply_image(image)
            new_size = image.shape[:2]
            centers = self.resize_transform_clip.apply_coords_torch(centers, ori_size)
            masks = F.interpolate(masks.float(), (new_size[0], new_size[1]),
                                  mode="bilinear", align_corners=False)  # N, 1, H, W
            bit_masks = masks > self.mask_threshold
            boxes = self.resize_transform_clip.apply_boxes_torch(boxes, ori_size)
            boxes_centers, axes_lengths = boxes_to_circles(boxes, *new_size)
            clip_inputs = []
            for center, mask, box, boxes_center, axes_length in zip(centers, bit_masks, boxes, boxes_centers, axes_lengths):
                res = image.copy()
                center = center.cpu().numpy()
                mask = mask.squeeze(0).cpu().numpy()
                box = box.int().cpu().numpy()
                boxes_center = boxes_center.cpu().numpy()
                axes_length = axes_length.cpu().numpy()
                if 'mask' == visual_prompt:
                    overlay = res.copy()
                    overlay[mask == 1] = np.array(str2rgb(self.color_mask))
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                if 'grayscale_mask' == visual_prompt:
                    gray = cv2.cvtColor(res.copy(), cv2.COLOR_BGR2GRAY)[:, :, None].repeat(3, -1)
                    res[mask == 0] = gray[mask == 0]
                if 'reverse_mask' == visual_prompt:
                    overlay = res.copy()
                    overlay[mask == 0] = np.array(str2rgb(self.color_mask))
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                if 'blur_mask' == visual_prompt:
                    res = Image.fromarray(res)
                    overlay = res.copy()
                    overlay = overlay.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    overlay.paste(res, mask=Image.fromarray(scale_mask(mask, self.contour_scale)))
                    res = np.array(overlay)
                if 'contour' == visual_prompt:
                    contours, hierarchy = cv2.findContours(mask.astype(
                        np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # areas = [cv2.contourArea(c) for c in contours]
                    # max_index = np.argmax(areas)
                    # cnt = contours[max_index]
                    if self.contour_scale == 1:
                        scaled_contours = contours
                    else:
                        scaled_contours = [scale_contour(cnt, self.contour_scale) for cnt in contours]
                    res = cv2.drawContours(res, scaled_contours, contourIdx=-1,
                                           color=str2rgb(self.color_line), thickness=self.thickness)
                if 'keypoint' == visual_prompt:
                    res = cv2.circle(res, center.astype(int), int(0.06 * self.clip_image_size),
                                     color=str2rgb(self.color_line), thickness=self.thickness)
                if 'circle_mask' == visual_prompt:
                    overlay = res.copy()
                    overlay = cv2.ellipse(overlay, boxes_center.astype(int), axes_length.astype(int), 0, 0, 360,
                                          color=str2rgb(self.color_mask), thickness=-1)
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                if 'grayscale_circle_mask' == visual_prompt:
                    circle_mask = np.zeros(res.shape, dtype=np.uint8)
                    circle_mask = cv2.ellipse(circle_mask, boxes_center.astype(int), axes_length.astype(int), 0, 0, 360,
                                              color=(255, 255, 255), thickness=-1)[:, :, 0]
                    gray = cv2.cvtColor(res.copy(), cv2.COLOR_BGR2GRAY)[:, :, None].repeat(3, -1)
                    res[circle_mask == 0] = gray[circle_mask == 0]
                if 'reverse_circle_mask' == visual_prompt:
                    circle_mask = np.zeros(res.shape, dtype=np.uint8)
                    circle_mask = cv2.ellipse(circle_mask, boxes_center.astype(int), axes_length.astype(int), 0, 0, 360,
                                              color=(255, 255, 255), thickness=-1)[:, :, 0]
                    overlay = res.copy()
                    overlay[circle_mask == 0] = np.array(str2rgb(self.color_mask))
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                if 'blur_circle_mask' == visual_prompt:
                    circle_mask = np.zeros(res.shape, dtype=np.uint8)
                    circle_mask = cv2.ellipse(circle_mask, boxes_center.astype(int), axes_length.astype(int), 0, 0, 360,
                                              color=(255, 255, 255), thickness=-1)[:, :, 0]
                    res = Image.fromarray(res)
                    overlay = res.copy()
                    overlay = overlay.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    overlay.paste(res, mask=Image.fromarray(scale_mask(circle_mask, self.contour_scale)))
                    res = np.array(overlay)
                if 'circle' == visual_prompt:
                    res = cv2.ellipse(res, boxes_center.astype(int), axes_length.astype(int), 0, 0, 360,
                                      color=str2rgb(self.color_line), thickness=self.thickness)
                if 'box_mask' == visual_prompt:
                    overlay = res.copy()
                    overlay = cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]),
                                            color=str2rgb(self.color_mask), thickness=-1)
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                if 'grayscale_box_mask' == visual_prompt:
                    box_mask = np.zeros(res.shape, dtype=np.uint8)
                    box_mask = cv2.rectangle(box_mask, (box[0], box[1]), (box[2], box[3]),
                                             color=(255, 255, 255), thickness=-1)[:, :, 0]
                    gray = cv2.cvtColor(res.copy(), cv2.COLOR_BGR2GRAY)[:, :, None].repeat(3, -1)
                    res[box_mask == 0] = gray[box_mask == 0]
                if 'reverse_box_mask' == visual_prompt:
                    box_mask = np.zeros(res.shape, dtype=np.uint8)
                    box_mask = cv2.rectangle(box_mask, (box[0], box[1]), (box[2], box[3]),
                                             color=(255, 255, 255), thickness=-1)[:, :, 0]
                    overlay = res.copy()
                    overlay[box_mask == 0] = np.array(str2rgb(self.color_mask))
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                if 'blur_box_mask' == visual_prompt:
                    box_mask = np.zeros(res.shape, dtype=np.uint8)
                    box_mask = cv2.rectangle(box_mask, (box[0], box[1]), (box[2], box[3]),
                                             color=(255, 255, 255), thickness=-1)[:, :, 0]
                    res = Image.fromarray(res)
                    overlay = res.copy()
                    overlay = overlay.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    overlay.paste(res, mask=Image.fromarray(box_mask))
                    res = np.array(overlay)
                if 'box' == visual_prompt:
                    res = cv2.rectangle(res, (box[0], box[1]), (box[2], box[3]),
                                        color=str2rgb(self.color_line), thickness=self.thickness)
                clip_inputs.append(res)

            clip_inputs = np.stack(clip_inputs)
            clip_inputs = torch.from_numpy(clip_inputs).float().permute(0, 3, 1, 2).to(self.device)
            # rgb -> normalized
            clip_inputs = (clip_inputs - self.pixel_mean) / self.pixel_std
            if self.clip_processing == 'padding':
                # Pad
                h, w = clip_inputs.shape[-2:]
                padh = self.clip_image_size - h
                padw = self.clip_image_size - w
                clip_inputs = F.pad(clip_inputs, (0, padw, 0, padh))
                # clip_inputs = F.pad(clip_inputs, (math.floor(padw / 2), math.ceil(padw / 2),
                #                     math.floor(padh / 2), math.ceil(padh / 2)))
            elif self.clip_processing == 'center_crop':
                # center crop
                clip_inputs = TF.center_crop(clip_inputs, (self.clip_image_size, self.clip_image_size))
            elif self.clip_processing == 'bias_crop':
                clip_inputs = torch.cat([bias_crop(c.unsqueeze(0), (self.clip_image_size, self.clip_image_size), b)
                                        for c, b in zip(clip_inputs, boxes)], 0)
            elif self.clip_processing == 'resize':
                clip_inputs = F.interpolate(clip_inputs, (self.clip_image_size, self.clip_image_size),
                                            mode='bilinear', align_corners=False)
            else:
                raise NotImplementedError

        # return nomalized inputs (N, 3, H, W)
        return clip_inputs
