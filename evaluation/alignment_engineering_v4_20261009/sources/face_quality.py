"""CPU-only image QC and approximate image-space measurements, not diagnosis."""
import argparse
import hashlib
import itertools
import json
import pathlib
import time

import cv2
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAMS = {
    "scale_factor": 1.05, "min_neighbors": 5, "min_face_px": 80,
    "min_laplacian_variance": 50, "soft_blur_variance": 170,
    "max_clipped_fraction": 0.30, "max_monochrome_fraction": 0.98,
    "detail_warning_face_px": 200,
}


def eye_pair(eyes, face_width, face_height):
    pairs = []
    for first, second in itertools.combinations(eyes, 2):
        a, b = sorted([first, second], key=lambda r: r[0])
        ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
        bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
        distance = bx - ax
        if (0.20 * face_width <= distance <= 0.75 * face_width
                and abs(ay - by) <= 0.15 * face_height
                and ax < 0.50 * face_width < bx):
            pairs.append((abs(ay - by), a, b))
    if not pairs:
        return None
    _, a, b = min(pairs, key=lambda p: p[0])
    return [list(map(int, a)), list(map(int, b))]


def rectangle(face, relative):
    x, y, w, h = face
    x0, y0, x1, y1 = relative
    return [x + round(w*x0), y + round(h*y0),
            x + round(w*x1), y + round(h*y1)]


def patch_stats(rgb, bbox):
    x0, y0, x1, y1 = bbox
    patch = rgb[y0:y1, x0:x1]
    lab = cv2.cvtColor(patch.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    grey = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return {
        "bbox_xyxy": bbox, "pixel_count": int(patch.shape[0]*patch.shape[1]),
        "rgb_median": [round(float(v), 3) for v in np.median(patch.reshape(-1, 3), axis=0)],
        "lab_median": [round(float(v), 3) for v in np.median(lab.reshape(-1, 3), axis=0)],
        "grey_mean": round(float(grey.mean()), 3),
        "clipped_high_fraction": round(float((patch.max(axis=2) >= 250).mean()), 4),
        "clipped_low_fraction": round(float((patch.min(axis=2) <= 5).mean()), 4),
    }


def analyze(path, reference_bbox=None):
    start = time.perf_counter()
    result = {
        "image_present": bool(path), "measurement_type": "image_space_only",
        "not_a_medical_assessment": True, "allow_medical_colour_inference": False,
        "coordinate_frame": "displayed image left/right; anatomical side and mirroring unknown",
        "parameters": PARAMS, "warnings": [], "reject_reasons": [],
        "patches": {}, "detection_candidates": [],
    }
    if not path:
        result.update(usable_for_technical_description=False,
                      reject_reasons=["no_image_received"], cv_latency_sec=0.0)
        return result
    cv2.setNumThreads(4)
    source = pathlib.Path(path)
    with Image.open(source) as im:
        orientation = int(im.getexif().get(274, 1))
        result["icc_profile_present"] = bool(im.info.get("icc_profile"))
        rgb = np.asarray(im.convert("RGB"))
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    result.update(image_path=str(source), image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  width=int(rgb.shape[1]), height=int(rgb.shape[0]), exif_orientation=orientation)
    if orientation != 1:
        result["reject_reasons"].append("exif_orientation_requires_consistent_preprocessing")
    result["monochrome_fraction"] = round(float((np.ptp(rgb.astype(np.int16), axis=2) <= 3).mean()), 4)
    face_model = pathlib.Path(cv2.data.haarcascades)/"haarcascade_frontalface_default.xml"
    eye_model = pathlib.Path(cv2.data.haarcascades)/"haarcascade_eye_tree_eyeglasses.xml"
    result["detector"] = {
        "opencv_version": cv2.__version__, "type": "Haar frontal face plus paired eyes",
        "face_xml_sha256": hashlib.sha256(face_model.read_bytes()).hexdigest(),
        "eye_xml_sha256": hashlib.sha256(eye_model.read_bytes()).hexdigest(),
    }
    faces = cv2.CascadeClassifier(str(face_model)).detectMultiScale(
        grey, scaleFactor=PARAMS["scale_factor"], minNeighbors=PARAMS["min_neighbors"], minSize=(40, 40))
    eye_detector = cv2.CascadeClassifier(str(eye_model))
    verified = []
    for x, y, w, h in faces:
        candidates = eye_detector.detectMultiScale(
            grey[y:y+int(h*.65), x:x+w], 1.05, 3, minSize=(10, 10))
        pair = eye_pair(candidates, w, h)
        bbox = list(map(int, [x, y, w, h]))
        result["detection_candidates"].append({"bbox_xywh": bbox, "paired_eyes": pair})
        if pair:
            verified.append(bbox)
    result["raw_face_count"] = len(faces)
    result["eye_verified_count"] = len(verified)
    if result["monochrome_fraction"] >= PARAMS["max_monochrome_fraction"]:
        result["reject_reasons"].append("nearly_monochrome_colour_assessment_not_supported")
    if len(verified) != 1:
        result["reject_reasons"].append("single_frontal_region_not_confirmed")
    else:
        bbox = verified[0]
        result["detected_face_bbox_xywh"] = bbox
        x, y, w, h = bbox
        roi = grey[y:y+h, x:x+w]
        result["laplacian_variance"] = round(float(cv2.Laplacian(roi, cv2.CV_64F).var()), 3)
        result["face_high_clip_fraction"] = round(float((roi >= 245).mean()), 4)
        result["face_low_clip_fraction"] = round(float((roi <= 10).mean()), 4)
        if min(w, h) < PARAMS["min_face_px"]:
            result["reject_reasons"].append("localized_region_too_small")
        if result["laplacian_variance"] < PARAMS["min_laplacian_variance"]:
            result["reject_reasons"].append("very_low_sharpness")
        elif result["laplacian_variance"] < PARAMS["soft_blur_variance"]:
            result["warnings"].append("lower_sharpness")
        if max(result["face_high_clip_fraction"], result["face_low_clip_fraction"]) > PARAMS["max_clipped_fraction"]:
            result["reject_reasons"].append("substantial_exposure_clipping")
        if min(w, h) < PARAMS["detail_warning_face_px"]:
            result["warnings"].append("low_resolution_for_fine_facial_features")
        measurement_bbox = reference_bbox or bbox
        result["measurement_bbox_xywh"] = list(measurement_bbox)
        result["fixed_reference_roi_used"] = reference_bbox is not None
        for key, relative in [("image_left_patch", (.10, .52, .34, .75)),
                              ("image_right_patch", (.66, .52, .90, .75))]:
            box = rectangle(measurement_bbox, relative)
            if 0 <= box[0] < box[2] <= rgb.shape[1] and 0 <= box[1] < box[3] <= rgb.shape[0]:
                result["patches"][key] = patch_stats(rgb, box)
                if result["patches"][key]["clipped_high_fraction"] > PARAMS["max_clipped_fraction"]:
                    result["reject_reasons"].append(key+"_substantial_channel_clipping")
            else:
                result["reject_reasons"].append("measurement_roi_out_of_bounds")
    result["warnings"].extend([
        "no_colour_card_or_capture_calibration", "patches_are_geometric_not_segmented_skin",
        "pose_occlusion_and_mirroring_not_verified", "RGB_is_assumed_sRGB_not_physically_calibrated",
    ])
    result["usable_for_technical_description"] = not result["reject_reasons"]
    result["cv_latency_sec"] = time.perf_counter()-start
    return result


def annotate(path, result, output):
    image = cv2.imread(str(path))
    for candidate in result["detection_candidates"]:
        x, y, w, h = candidate["bbox_xywh"]
        colour = (0, 180, 0) if candidate["paired_eyes"] else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), colour, 1)
    for name, stats in result["patches"].items():
        x0, y0, x1, y1 = stats["bbox_xyxy"]
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 120, 0), 1)
    cv2.imwrite(str(output), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.image)
    pathlib.Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
