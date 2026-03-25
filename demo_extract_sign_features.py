"""Demo ekstrakcji cech z nagrania PJM/ASL z użyciem MediaPipe.

Co robi skrypt:
1. Wybiera nagranie na podstawie VIDEO_ID z pliku CSV.
2. Przetwarza klatki przez MediaPipe Holistic.
3. Zapisuje cechy klatka-po-klatce do CSV (gotowe jako wejście do modelu sekwencyjnego).
4. Drukuje podsumowanie i przykładowe rekordy.

Wymagania:
    pip install opencv-python mediapipe

Uruchomienie:
    python demo_extract_sign_features.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Optional

import cv2
import mediapipe as mp


# =============================
# Konfiguracja
# =============================
CSV_PATH = Path("merged_datasets/universal_metadata_has_video_no_anomalies.csv")
VIDEO_ID = "31220"
SOURCE_FILTER: Optional[str] = "signingsavvy"
OUTPUT_DIR = Path("outputs")
MAX_FRAMES: Optional[int] = 120  # None = bez limitu
SHOW_PREVIEW = True
SAVE_ANNOTATED_VIDEO = True
PREVIEW_WINDOW = "Feature extraction preview"


POSE_LM_COUNT = 33
HAND_LM_COUNT = 21


@dataclass
class VideoRecord:
    label: str
    source: str
    video_path: Path
    width: int
    height: int


def normalize_video_id(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("VIDEO_ID nie może być pusty")
    if value.isdigit():
        return str(int(value))
    return value


def parse_dim(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def get_video_id_from_path(video_path: str) -> str:
    stem = PureWindowsPath(video_path).stem
    return normalize_video_id(stem)


def load_record(csv_path: Path, video_id: str, source_filter: Optional[str]) -> VideoRecord:
    if not csv_path.exists():
        raise FileNotFoundError(f"Nie znaleziono CSV: {csv_path}")

    wanted_id = normalize_video_id(video_id)
    source_filter_lc = source_filter.lower().strip() if source_filter else None

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue

            row_video_path = row.get("video_path", "")
            if not row_video_path:
                continue

            if get_video_id_from_path(row_video_path) != wanted_id:
                continue

            row_source = str(row.get("source", ""))
            if source_filter_lc and row_source.lower().strip() != source_filter_lc:
                continue

            return VideoRecord(
                label=str(row.get("label", "")),
                source=row_source,
                video_path=Path(row_video_path),
                width=parse_dim(row.get("video_width", "0")),
                height=parse_dim(row.get("video_height", "0")),
            )

    raise LookupError(f"Nie znaleziono rekordu dla VIDEO_ID={video_id}")


def flatten_pose(pose_landmarks) -> list[float]:
    values: list[float] = []
    if pose_landmarks is None:
        # x,y,z,visibility dla 33 punktów
        return [0.0] * (POSE_LM_COUNT * 4)

    for lm in pose_landmarks.landmark:
        values.extend([lm.x, lm.y, lm.z, lm.visibility])
    return values


def flatten_hand(hand_landmarks) -> list[float]:
    values: list[float] = []
    if hand_landmarks is None:
        # x,y,z dla 21 punktów
        return [0.0] * (HAND_LM_COUNT * 3)

    for lm in hand_landmarks.landmark:
        values.extend([lm.x, lm.y, lm.z])
    return values


def build_feature_names() -> list[str]:
    names: list[str] = []

    for i in range(POSE_LM_COUNT):
        names.extend(
            [
                f"pose_{i}_x",
                f"pose_{i}_y",
                f"pose_{i}_z",
                f"pose_{i}_v",
            ]
        )

    for i in range(HAND_LM_COUNT):
        names.extend(
            [
                f"left_hand_{i}_x",
                f"left_hand_{i}_y",
                f"left_hand_{i}_z",
            ]
        )

    for i in range(HAND_LM_COUNT):
        names.extend(
            [
                f"right_hand_{i}_x",
                f"right_hand_{i}_y",
                f"right_hand_{i}_z",
            ]
        )

    return names


def extract_features(
    record: VideoRecord,
    output_dir: Path,
    max_frames: Optional[int],
    show_preview: bool,
    save_annotated_video: bool,
) -> tuple[Path, Path, int, int]:
    if not record.video_path.exists():
        raise FileNotFoundError(f"Brak pliku wideo: {record.video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = record.video_path.stem
    output_csv = output_dir / f"features_{video_stem}.csv"
    output_schema = output_dir / f"features_{video_stem}_schema.json"
    annotated_video_path = output_dir / f"annotated_{video_stem}.mp4"

    feature_names = build_feature_names()

    header = [
        "frame_idx",
        "time_sec",
        "label",
        "source",
        "video_file",
        "video_width",
        "video_height",
    ] + feature_names

    cap = cv2.VideoCapture(str(record.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie udało się otworzyć wideo: {record.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    video_writer = None
    if save_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = record.width if record.width > 0 else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = record.height if record.height > 0 else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if out_w <= 0:
            out_w = 640
        if out_h <= 0:
            out_h = 480
        video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (out_w, out_h))

    if show_preview:
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

    frame_idx = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerow(header)

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                if max_frames is not None and frame_idx > max_frames:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                row_features = []
                row_features.extend(flatten_pose(results.pose_landmarks))
                row_features.extend(flatten_hand(results.left_hand_landmarks))
                row_features.extend(flatten_hand(results.right_hand_landmarks))

                row = [
                    frame_idx,
                    frame_idx / fps,
                    record.label,
                    record.source,
                    record.video_path.name,
                    record.width,
                    record.height,
                ] + row_features

                csv_writer.writerow(row)

                # Wizualizacja landmarkow na obrazie
                overlay_frame = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        overlay_frame,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                    )

                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        overlay_frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                    )

                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        overlay_frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                    )

                status = f"{record.video_path.name} | frame={frame_idx} | label={record.label}"
                cv2.putText(overlay_frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                if video_writer is not None:
                    video_writer.write(overlay_frame)

                if show_preview:
                    cv2.imshow(PREVIEW_WINDOW, overlay_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if show_preview:
        cv2.destroyWindow(PREVIEW_WINDOW)

    schema = {
        "video_id": video_stem,
        "feature_vector_length": len(feature_names),
        "blocks": {
            "pose": {
                "landmarks": POSE_LM_COUNT,
                "dims_per_landmark": ["x", "y", "z", "visibility"],
                "length": POSE_LM_COUNT * 4,
            },
            "left_hand": {
                "landmarks": HAND_LM_COUNT,
                "dims_per_landmark": ["x", "y", "z"],
                "length": HAND_LM_COUNT * 3,
            },
            "right_hand": {
                "landmarks": HAND_LM_COUNT,
                "dims_per_landmark": ["x", "y", "z"],
                "length": HAND_LM_COUNT * 3,
            },
        },
        "csv_path": str(output_csv),
    }

    with output_schema.open("w", encoding="utf-8") as f_schema:
        json.dump(schema, f_schema, indent=2, ensure_ascii=True)

    return output_csv, output_schema, annotated_video_path, frame_idx


def print_preview(csv_path: Path, rows_to_show: int = 2) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        print("\nPrzykladowe rekordy cech:")
        shown = 0
        for row in reader:
            sample = {
                "frame_idx": row["frame_idx"],
                "time_sec": row["time_sec"],
                "label": row["label"],
                "source": row["source"],
                "pose_0_x": row["pose_0_x"],
                "pose_0_y": row["pose_0_y"],
                "left_hand_0_x": row["left_hand_0_x"],
                "left_hand_0_y": row["left_hand_0_y"],
                "right_hand_0_x": row["right_hand_0_x"],
                "right_hand_0_y": row["right_hand_0_y"],
            }
            print(json.dumps(sample, ensure_ascii=True))
            shown += 1
            if shown >= rows_to_show:
                break


def main() -> None:
    record = load_record(CSV_PATH, VIDEO_ID, SOURCE_FILTER)

    print("Wybrane nagranie:")
    print(f"  plik: {record.video_path}")
    print(f"  etykieta: {record.label}")
    print(f"  zrodlo: {record.source}")
    print(f"  rozdzielczosc: {record.width}x{record.height}")

    output_csv, output_schema, annotated_video, frames = extract_features(
        record,
        OUTPUT_DIR,
        MAX_FRAMES,
        show_preview=SHOW_PREVIEW,
        save_annotated_video=SAVE_ANNOTATED_VIDEO,
    )

    print("\nZapisano:")
    print(f"  cechy: {output_csv}")
    print(f"  schemat: {output_schema}")
    if SAVE_ANNOTATED_VIDEO:
        print(f"  video z punktami: {annotated_video}")
    print(f"  liczba klatek: {frames}")

    print_preview(output_csv, rows_to_show=3)


if __name__ == "__main__":
    main()
