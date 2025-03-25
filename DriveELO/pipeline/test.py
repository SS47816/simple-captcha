import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
    fcos_resnet50_fpn,
    maskrcnn_resnet50_fpn
)
from datasets import load_dataset
from torchvision.transforms import functional as F
from tqdm import tqdm
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json
from PIL import Image

# -----------------------------
# CONFIGURATION
# -----------------------------
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD COCO 2017 DATASET (Hugging Face API)
# -----------------------------
# coco_dataset = load_dataset("imagefolder", data_dir="huggingface/coco/2017")
coco_dataset = load_dataset("phiyodr/coco2017")
# Extract validation set
val_data = coco_dataset["validation"]

print(val_data[0])

# -----------------------------
# PRETRAINED MODELS TO TEST
# -----------------------------
models = {
    "FasterRCNN_ResNet50": fasterrcnn_resnet50_fpn(weights="COCO_V1").eval().to(DEVICE),
    "RetinaNet_ResNet50": retinanet_resnet50_fpn(weights="COCO_V1").eval().to(DEVICE),
    "SSD_VGG16": ssd300_vgg16(weights="COCO_V1").eval().to(DEVICE),
    "FCOS_ResNet50": fcos_resnet50_fpn(weights="COCO_V1").eval().to(DEVICE),
    "MaskRCNN_ResNet50": maskrcnn_resnet50_fpn(weights="COCO_V1").eval().to(DEVICE),
}

# -----------------------------
# RUN INFERENCE AND SAVE RESULTS
# -----------------------------
def evaluate_model(model, model_name):
    print(f"\nEvaluating {model_name}...\n")
    results = []

    with torch.no_grad():
        for sample in tqdm(val_data):
            image = Image.open(sample["image_path"]).convert("RGB")
            image_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)

            # Run inference
            output = model(image_tensor)[0]

            image_id = sample["annotation"][0]["image_id"] if len(sample["annotation"]) > 0 else -1
            if image_id == -1:
                continue

            for j in range(len(output["boxes"])):
                bbox = output["boxes"][j].tolist()
                score = output["scores"][j].item()
                label = int(output["labels"][j].item())

                results.append({
                    "image_id": int(image_id),
                    "category_id": label,
                    "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # Convert to COCO format
                    "score": float(score)
                })

    # Save JSON results
    results_file = f"coco_results_{model_name}.json"
    with open(results_file, "w") as f:
        json.dump(results, f)

    # Evaluate using COCO API
    coco_gt = coco.COCO("huggingface/coco/2017/annotations/instances_val2017.json")
    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# Run evaluation on each model
for model_name, model in models.items():
    evaluate_model(model, model_name)
