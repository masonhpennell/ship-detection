import os
import cv2
import numpy as np
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

IMAGE_DIR = "images"
MASK_DIR = "masks"
os.makedirs(MASK_DIR, exist_ok=True)

CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=48,
    pred_iou_thresh=0.90,
    stability_score_thresh=0.95,
    min_mask_region_area=1000
)

for fname in tqdm(os.listdir(IMAGE_DIR)):
    path = os.path.join(IMAGE_DIR, fname)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(img)

    # choose largest non-background region
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    combined = np.zeros(img.shape[:2], dtype=np.uint8)

    for m in masks[:5]:
        seg = m["segmentation"].astype(np.uint8)
        combined = np.maximum(combined, seg)
    


    out = (combined * 255).astype(np.uint8)

    out_name = os.path.splitext(fname)[0] + ".png"
    cv2.imwrite(os.path.join(MASK_DIR, out_name), out)