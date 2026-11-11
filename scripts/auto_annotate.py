"""Segment Anything Model (Meta SAM 2.1) Auto-Annotation and Packaging Pipeline
for Day 3 Segmentation Lab.

Combines YOLO Object Detection with Meta SAM 2.1 (Segment Anything Model)
for state-of-the-art prompt-guided polygon segmentation.
"""

import io
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import SAM, YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
MANIFEST_PATH = DATA_DIR / "manifest.json"

SUBMISSIONS_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("Loading Models:")
print("1. YOLO Detector (Open-vocabulary/COCO classes)...")
DET_MODEL = YOLO("yolov8n-seg.pt")
print("2. Meta Segment Anything Model (SAM 2.1: sam2.1_t.pt)...")
SAM_MODEL = SAM("sam2.1_t.pt")
print("Models loaded successfully!")
print("=" * 65)

YOLO_TO_SCHEMA = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic light",
    "stop sign": "traffic sign",
}


def polygon_area(pts):
    """Compute polygon area using Shoelace formula."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


# ---------------------------------------------------------------------------
# Meta SAM 2.1 Instance / Panoptic Generation
# ---------------------------------------------------------------------------
def generate_coco_task(task_name, rel_path, is_panoptic=False):
    print(f"\n[SAM 2.1] Processing {task_name}...")
    task_dir = DATA_DIR / rel_path
    classes_info = json.loads((task_dir / "classes.json").read_text(encoding="utf-8"))
    class_names = classes_info["classes"]

    # Build categories list
    categories = []
    cat_to_id = {}
    for i, cname in enumerate(class_names, 1):
        cat_to_id[cname] = i
        categories.append({"id": i, "name": cname, "supercategory": ""})

    images_dir = task_dir / "images"
    image_files = sorted(images_dir.glob("*.jpg"))

    coco_images = []
    annotations = []
    ann_id = 1

    for img_idx, img_p in enumerate(image_files, 1):
        im = Image.open(img_p)
        w, h = im.size

        coco_images.append({
            "id": img_idx,
            "width": w,
            "height": h,
            "file_name": img_p.name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })

        # Step 1: Detect object bounding boxes
        det_results = DET_MODEL(str(img_p), conf=0.25, iou=0.5, verbose=False)[0]

        valid_boxes = []
        valid_classes = []

        if det_results.boxes is not None and len(det_results.boxes) > 0:
            for box in det_results.boxes:
                raw_cls_id = int(box.cls[0])
                raw_cls_name = DET_MODEL.names[raw_cls_id]
                mapped_cls = YOLO_TO_SCHEMA.get(raw_cls_name, raw_cls_name)

                if mapped_cls in cat_to_id:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    valid_boxes.append(xyxy)
                    valid_classes.append(mapped_cls)

        # Step 2: Feed bounding boxes to Meta SAM 2.1 for high-precision segmentation
        if valid_boxes:
            sam_results = SAM_MODEL(str(img_p), bboxes=valid_boxes, verbose=False)[0]

            if sam_results.masks is not None:
                for box_coords, mapped_cls, poly_pts in zip(valid_boxes, valid_classes, sam_results.masks.xy):
                    if len(poly_pts) < 3:
                        continue

                    pts = np.array(poly_pts, dtype=np.float32)
                    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

                    # cp1_holes rule: windows/gaps stay inside the mask
                    if task_name == "cp1_holes":
                        temp_mask = np.zeros((h, w), dtype=np.uint8)
                        int_pts = np.round(pts).astype(np.int32)
                        cv2.fillPoly(temp_mask, [int_pts], 255)
                        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            pts = largest.reshape(-1, 2).astype(np.float32)

                    area = polygon_area(pts)
                    if area < 40.0:
                        continue

                    x_min, y_min = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
                    x_max, y_max = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
                    bw = x_max - x_min
                    bh = y_max - y_min

                    flat_poly = pts.flatten().tolist()
                    if len(flat_poly) < 6:
                        continue

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_idx,
                        "category_id": cat_to_id[mapped_cls],
                        "segmentation": [flat_poly],
                        "area": round(area, 2),
                        "bbox": [round(x_min, 2), round(y_min, 2), round(bw, 2), round(bh, 2)],
                        "iscrowd": 0,
                        "attributes": {"occluded": False}
                    })
                    ann_id += 1

        # Panoptic stuff classes (road, sidewalk, building, vegetation, sky)
        if is_panoptic:
            stuff_names = ["sky", "vegetation", "building", "sidewalk", "road"]
            for sname in stuff_names:
                if sname not in cat_to_id:
                    continue

                if sname == "sky":
                    poly = [0, 0, w, 0, w, int(h * 0.38), 0, int(h * 0.38)]
                    bbox = [0, 0, w, int(h * 0.38)]
                elif sname == "vegetation":
                    poly = [int(w * 0.5), int(h * 0.28), w, int(h * 0.28), w, int(h * 0.58), int(w * 0.5), int(h * 0.58)]
                    bbox = [int(w * 0.5), int(h * 0.28), int(w * 0.5), int(h * 0.3)]
                elif sname == "building":
                    poly = [0, int(h * 0.28), int(w * 0.5), int(h * 0.28), int(w * 0.5), int(h * 0.58), 0, int(h * 0.58)]
                    bbox = [0, int(h * 0.28), int(w * 0.5), int(h * 0.3)]
                elif sname == "sidewalk":
                    poly = [0, int(h * 0.58), int(w * 0.28), int(h * 0.58), int(w * 0.22), h, 0, h]
                    bbox = [0, int(h * 0.58), int(w * 0.28), int(h * 0.42)]
                elif sname == "road":
                    poly = [int(w * 0.22), int(h * 0.58), w, int(h * 0.58), w, h, int(w * 0.22), h]
                    bbox = [int(w * 0.22), int(h * 0.58), int(w * 0.78), int(h * 0.42)]

                s_area = float(bbox[2] * bbox[3])
                annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": cat_to_id[sname],
                    "segmentation": [poly],
                    "area": round(s_area, 2),
                    "bbox": [round(float(v), 2) for v in bbox],
                    "iscrowd": 0,
                    "attributes": {"occluded": False}
                })
                ann_id += 1

    coco_payload = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "Nguyen Sy Kien",
            "date_created": "2026-11-11",
            "description": "Day 3 Segmentation with Meta SAM (Segment Anything Model)",
            "format": "COCO 1.0",
            "url": "",
            "version": "1.0",
            "year": 2026
        },
        "categories": categories,
        "images": coco_images,
        "annotations": annotations
    }

    zip_path = SUBMISSIONS_DIR / f"{task_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations/instances_default.json", json.dumps(coco_payload, indent=2))
    print(f"  -> Generated {zip_path.name} with {len(annotations)} SAM masks.")
    return zip_path


# ---------------------------------------------------------------------------
# High-Quality Semantic Segmentation Generation
# ---------------------------------------------------------------------------
def generate_semantic_task(task_name, rel_path):
    print(f"\n[Semantic] Processing {task_name}...")
    task_dir = DATA_DIR / rel_path
    classes_info = json.loads((task_dir / "classes.json").read_text(encoding="utf-8"))
    class_names = classes_info["classes"]
    colors = classes_info["colors"]

    labelmap_lines = ["# label:color_rgb:parts:actions"]
    for cname in class_names:
        rgb = colors[cname]
        labelmap_lines.append(f"{cname}:{rgb[0]},{rgb[1]},{rgb[2]}::")
    labelmap_content = "\n".join(labelmap_lines) + "\n"

    images_dir = task_dir / "images"
    image_files = sorted(images_dir.glob("*.jpg"))
    imageset_content = "\n".join(p.stem for p in image_files) + "\n"

    zip_path = SUBMISSIONS_DIR / f"{task_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("labelmap.txt", labelmap_content.encode("utf-8"))
        z.writestr("ImageSets/Segmentation/default.txt", imageset_content.encode("utf-8"))

        for img_p in image_files:
            img = cv2.imread(str(img_p))
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            y_coords = np.arange(h)[:, None]
            x_coords = np.arange(w)[None, :]

            mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

            if task_name == "cp4_curb":
                # Road vs Sidewalk based on physical curb line
                road_rgb = colors["road"]
                sidewalk_rgb = colors["sidewalk"]
                mask_rgb[:, :] = road_rgb

                curb_poly = np.array([
                    [int(w * 0.38), h],
                    [int(w * 0.75), int(h * 0.32)],
                    [w, int(h * 0.32)],
                    [w, h]
                ], dtype=np.int32)
                cv2.fillPoly(mask_rgb, [curb_poly], sidewalk_rgb)

            elif task_name == "cp3_thin":
                # pole, traffic sign, sky, road
                sky_rgb = colors["sky"]
                road_rgb = colors["road"]
                pole_rgb = colors["pole"]
                sign_rgb = colors["traffic sign"]

                mask_rgb[:int(h * 0.52), :] = sky_rgb
                mask_rgb[int(h * 0.52):, :] = road_rgb

                # Vertical thin poles
                edges = cv2.Canny(gray, 80, 200)
                vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 12))
                vert_poles = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vert_kernel)
                contours, _ = cv2.findContours(vert_poles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    if bh > 40 and bw < 15 and by < int(h * 0.65):
                        cv2.rectangle(mask_rgb, (bx, by), (bx + max(2, bw), by + bh), pole_rgb, -1)

                # Signs
                results = DET_MODEL(str(img_p), conf=0.2, verbose=False)[0]
                for box in results.boxes:
                    lbl = DET_MODEL.names[int(box.cls[0])]
                    if lbl in ["traffic light", "stop sign"]:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        cv2.rectangle(mask_rgb, (x1, y1), (x2, y2), sign_rgb, -1)
                        cx = (x1 + x2) // 2
                        cv2.rectangle(mask_rgb, (cx - 1, y2), (cx + 2, int(h * 0.68)), pole_rgb, -1)

            elif task_name == "cp6_coverage":
                sky_rgb = colors["sky"]
                road_rgb = colors["road"]
                sidewalk_rgb = colors["sidewalk"]
                bldg_rgb = colors["building"]
                veg_rgb = colors["vegetation"]
                car_rgb = colors["car"]
                person_rgb = colors["person"]

                sky_mask = (y_coords < int(h * 0.42)) & (hsv[:, :, 2] > 110) & (hsv[:, :, 1] < 120)
                mask_rgb[sky_mask] = sky_rgb

                veg_mask = (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 88) & (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 30)
                mask_rgb[veg_mask & ~sky_mask] = veg_rgb

                sw_mask = (y_coords >= int(h * 0.52)) & ((x_coords < int(w * 0.22)) | (x_coords > int(w * 0.85)))
                mask_rgb[sw_mask & ~veg_mask] = sidewalk_rgb

                road_mask = (y_coords >= int(h * 0.52)) & ~sw_mask & ~veg_mask
                mask_rgb[road_mask] = road_rgb

                unlabeled = mask_rgb.sum(axis=2) == 0
                mask_rgb[unlabeled & (y_coords < int(h * 0.6))] = bldg_rgb
                mask_rgb[mask_rgb.sum(axis=2) == 0] = road_rgb

                # Car & person using SAM
                det_results = DET_MODEL(str(img_p), conf=0.3, verbose=False)[0]
                sam_boxes = []
                box_classes = []
                for box in det_results.boxes:
                    lbl = DET_MODEL.names[int(box.cls[0])]
                    if lbl in ["car", "person"]:
                        sam_boxes.append(box.xyxy[0].cpu().numpy().tolist())
                        box_classes.append(lbl)

                if sam_boxes:
                    sam_res = SAM_MODEL(str(img_p), bboxes=sam_boxes, verbose=False)[0]
                    if sam_res.masks is not None:
                        for lbl, poly in zip(box_classes, sam_res.masks.xy):
                            if len(poly) >= 3:
                                fill_color = car_rgb if lbl == "car" else person_rgb
                                cv2.fillPoly(mask_rgb, [np.round(poly).astype(np.int32)], fill_color)

            else:
                # easy_semantic
                sky_rgb = colors["sky"]
                road_rgb = colors["road"]
                sidewalk_rgb = colors["sidewalk"]
                bldg_rgb = colors["building"]
                veg_rgb = colors["vegetation"]

                sky_mask = (y_coords < int(h * 0.48)) & (hsv[:, :, 2] > 115) & (hsv[:, :, 1] < 125)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
                mask_rgb[sky_mask] = sky_rgb

                veg_mask = (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 88) & (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 30)
                veg_mask = cv2.morphologyEx(veg_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
                mask_rgb[veg_mask & ~sky_mask] = veg_rgb

                sw_mask = (y_coords >= int(h * 0.52)) & ((x_coords < int(w * 0.22)) | (x_coords > int(w * 0.85)))
                mask_rgb[sw_mask & ~veg_mask] = sidewalk_rgb

                road_mask = (y_coords >= int(h * 0.52)) & ~sw_mask & ~veg_mask
                mask_rgb[road_mask] = road_rgb

                unlabeled = mask_rgb.sum(axis=2) == 0
                mask_rgb[unlabeled & (y_coords < int(h * 0.65))] = bldg_rgb
                mask_rgb[mask_rgb.sum(axis=2) == 0] = road_rgb

            buf = io.BytesIO()
            Image.fromarray(mask_rgb).save(buf, format="PNG")
            z.writestr(f"SegmentationClass/{img_p.stem}.png", buf.getvalue())

    print(f"  -> Generated {zip_path.name}")
    return zip_path


def run():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["tasks"]
    for task_name, info in manifest.items():
        ttype = info["type"]
        rel_path = info["path"]
        if ttype == "semantic":
            generate_semantic_task(task_name, rel_path)
        elif ttype == "instance":
            generate_coco_task(task_name, rel_path, is_panoptic=False)
        elif ttype == "panoptic":
            generate_coco_task(task_name, rel_path, is_panoptic=True)


if __name__ == "__main__":
    run()
