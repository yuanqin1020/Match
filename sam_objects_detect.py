
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pycocotools import mask as mask_utils


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



model_type = "vit_h"
image_path = ".../google_14.jpg"
ckp_path = ".../sam/sam_vit_h_4b8939.pth"

image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry[model_type](checkpoint=ckp_path).to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image_rgb)
print(f"masks numbers: {len(masks)}")
print(masks[0].keys()) # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box', 'embed'])
print(f"masks segmentation shape: {masks[0]['segmentation'].shape}")
print(f"masks embed shape: {masks[0]['embed'].shape}")


predictor = SamPredictor(sam)
predictor.set_image(image_rgb)
image_embedding = predictor.get_image_embedding().cpu().numpy()
print(f"predictor embed: {image_embedding}")



# print(f"segmentation shape: {mask.shape()}")
