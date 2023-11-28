import matplotlib.pyplot as plt
import numpy as np
import torch


def show_masks(fig, anns, color=None, alpha=0.35):
    if isinstance(anns, torch.Tensor):
        anns = anns.cpu().numpy()
    if len(anns.shape) == 2:
        anns = anns[None, :, :]
    anns = [ann for ann in anns]
    # anns = sorted(anns, key=lambda x: x.sum(), reverse=True)
    if len(anns) == 0:
        return fig
    fig.gca().set_autoscale_on(False)
    for i, m in enumerate(anns):
        img = np.ones((m.shape[0], m.shape[1], 3))
        if color is None:
            color_mask = np.random.random((1, 3)).tolist()[0]
        else:
            color_mask = color[i]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        fig.gca().imshow(np.dstack((img, m * alpha)))
    return fig


def show_points(fig, coords, marker_size=10, color='green'):
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    coords = coords.astype(np.int64)
    fig.gca().scatter(coords[:, 0], coords[:, 1], color=color, s=marker_size, edgecolor='white', linewidth=1.25)
    return fig


def show_box(fig, box, color='green', lw=1):
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy()
    box = box.astype(np.int64)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    fig.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=lw))
    return fig
