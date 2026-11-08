import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# MobileNetV2 is used as a pretrained deep-learning feature extractor.
# First run may download ImageNet weights from Keras.
def load_feature_model():
    import tensorflow as tf
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(224, 224, 3)
    )
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    return model, preprocess


def _read_sampled_frames(video_path, sample_fps=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    interval = max(1, int(round(fps / sample_fps)))

    frames, times, motions = [], [], []
    previous_gray = None
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % interval == 0:
            small = cv2.resize(frame, (224, 224))
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            motion = 0.0
            if previous_gray is not None:
                motion = float(np.mean(cv2.absdiff(gray, previous_gray)) / 255.0)
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            times.append(frame_index / fps)
            motions.append(motion)
            previous_gray = gray
        frame_index += 1

    cap.release()
    if not frames:
        raise ValueError("No frames were extracted from the video.")
    return np.asarray(frames), np.asarray(times), np.asarray(motions), duration, fps


def _score_segments(features, times, motions, duration, segment_seconds, novelty_weight=0.8):
    starts = np.arange(0, max(duration, segment_seconds), segment_seconds)
    raw = []
    for start in starts:
        end = min(start + segment_seconds, duration)
        indices = np.where((times >= start) & (times < end))[0]
        if len(indices) == 0:
            continue

        local_features = features[indices]
        # Feature novelty: how much the visual content differs within the segment.
        center = np.mean(local_features, axis=0)
        novelty = float(np.mean(np.linalg.norm(local_features - center, axis=1)))
        # Motion is a simple complementary cue, not a claim about semantic importance.
        motion = float(np.mean(motions[indices]))
        raw.append({
            "start": float(start),
            "end": float(end),
            "novelty": novelty,
            "motion": motion,
        })

    if not raw:
        raise ValueError("Could not form any video segments.")

    novelty_values = np.array([x["novelty"] for x in raw], dtype=np.float32)
    motion_values = np.array([x["motion"] for x in raw], dtype=np.float32)

    def normalize(values):
        span = float(values.max() - values.min())
        return (values - values.min()) / span if span > 1e-8 else np.zeros_like(values)

    novelty_norm = normalize(novelty_values)
    motion_norm = normalize(motion_values)
    novelty_weight = float(np.clip(novelty_weight, 0.0, 1.0))
    motion_weight = 1.0 - novelty_weight
    for i, item in enumerate(raw):
        item["score"] = float(novelty_weight * novelty_norm[i] + motion_weight * motion_norm[i])
    return raw


def _select_non_overlapping(segments, target_seconds):
    # Greedy ranking by importance, with no overlapping time intervals.
    ranked = sorted(segments, key=lambda x: x["score"], reverse=True)
    selected = []
    used_duration = 0.0
    for candidate in ranked:
        length = candidate["end"] - candidate["start"]
        if used_duration + length > target_seconds and selected:
            continue
        overlaps = any(
            not (candidate["end"] <= chosen["start"] or candidate["start"] >= chosen["end"])
            for chosen in selected
        )
        if not overlaps:
            selected.append(candidate)
            used_duration += length
        if used_duration >= target_seconds:
            break
    return sorted(selected, key=lambda x: x["start"])


def _write_summary(video_path, selected, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not reopen the input video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the output MP4 file.")

    for segment in selected:
        cap.set(cv2.CAP_PROP_POS_MSEC, segment["start"] * 1000)
        end_time = segment["end"]
        while True:
            current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if current_ms >= end_time * 1000:
                break
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)

    writer.release()
    cap.release()


def summarize_video(video_path, segment_seconds=5, target_seconds=20, sample_fps=1.0, output_dir="outputs", novelty_weight=0.8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, times, motions, duration, _ = _read_sampled_frames(video_path, sample_fps=sample_fps)
    model, preprocess = load_feature_model()

    batch = preprocess(frames.astype(np.float32))
    features = model.predict(batch, batch_size=16, verbose=0)
    features = np.asarray(features, dtype=np.float32)

    segments = _score_segments(features, times, motions, duration, segment_seconds, novelty_weight=novelty_weight)
    selected = _select_non_overlapping(segments, target_seconds)
    if not selected:
        raise ValueError("No segments were selected. Try a longer target summary duration.")

    summary_path = str(output_dir / "video_summary.mp4")
    _write_summary(video_path, selected, summary_path)

    result_segments = []
    for index, segment in enumerate(selected, start=1):
        mid = (segment["start"] + segment["end"]) / 2
        frame_index = int(np.argmin(np.abs(times - mid)))
        thumb_path = str(output_dir / f"thumbnail_{index}.jpg")
        Image.fromarray(frames[frame_index]).save(thumb_path, quality=90)
        result_segments.append({
            **segment,
            "thumbnail": thumb_path,
        })

    summary_duration = sum(x["end"] - x["start"] for x in selected)
    return {
        "summary_path": summary_path,
        "original_duration": duration,
        "summary_duration": summary_duration,
        "segments": result_segments,
        "novelty_weight": float(novelty_weight),
        "motion_weight": float(1.0 - novelty_weight),
    }
