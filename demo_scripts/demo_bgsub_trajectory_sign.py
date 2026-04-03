"""Demo alternatywne: Background Subtraction + trajektoria ruchu.

To podejscie:
1. Oddziela ruchomy obiekt od tla (MOG2).
2. Czyści maske morfologicznie.
3. Rysuje bbox i trajektorie centroidu ruchu.

Wymagania:
    pip install opencv-python numpy

Uruchomienie:
    python demo_bgsub_trajectory_sign.py
"""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Optional

import cv2
import numpy as np


# =============================
# Konfiguracja
# =============================
CSV_PATH = Path("merged_datasets/universal_metadata_has_video_no_anomalies.csv")
VIDEO_ID = "31220"
SOURCE_FILTER: Optional[str] = "signingsavvy"
PLAY_MODE = "resolution_sweep"  # "single" albo "resolution_sweep"
WINDOW_NAME = "BG Subtraction + Trajectory demo"
DISPLAY_SIZE = (960, 540)
EXTRA_SLEEP_SECONDS = 0.03
SWEEP_SOURCE_FILTER: Optional[str] = None
SWEEP_MAX_VIDEOS = 20

# Parametry segmentacji ruchu
MOG2_HISTORY = 300
MOG2_VAR_THRESHOLD = 36
MOG2_DETECT_SHADOWS = False
MIN_CONTOUR_AREA = 900
MORPH_KERNEL_SIZE = 5
TRAJECTORY_LENGTH = 40


@dataclass
class VideoRecord:
    label: str
    source: str
    video_path: Path
    video_width: int
    video_height: int


def normalize_video_id(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("VIDEO_ID nie moze byc pusty")
    if value.isdigit():
        return str(int(value))
    return value


def get_video_id_from_path(video_path: str) -> str:
    stem = PureWindowsPath(video_path).stem
    return normalize_video_id(stem)


def parse_dimension(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_video_record(csv_path: Path, video_id: str, source_filter: Optional[str]) -> VideoRecord:
    if not csv_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku CSV: {csv_path}")

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

            row_video_id = get_video_id_from_path(row_video_path)
            if row_video_id != wanted_id:
                continue

            row_source = str(row.get("source", ""))
            if source_filter_lc and row_source.lower().strip() != source_filter_lc:
                continue

            return VideoRecord(
                label=str(row.get("label", "")),
                source=row_source,
                video_path=Path(row_video_path),
                video_width=parse_dimension(row.get("video_width", "0")),
                video_height=parse_dimension(row.get("video_height", "0")),
            )

    hint = f" i zrodla '{source_filter}'" if source_filter else ""
    raise LookupError(f"Nie znaleziono rekordu dla VIDEO_ID='{video_id}'{hint}.")


def load_records_distinct_resolutions(
    csv_path: Path,
    source_filter: Optional[str],
    max_videos: int,
) -> list[VideoRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku CSV: {csv_path}")

    source_filter_lc = source_filter.lower().strip() if source_filter else None
    by_resolution: dict[tuple[int, int], VideoRecord] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue

            if str(row.get("has_video", "")).lower().strip() != "true":
                continue

            row_source = str(row.get("source", ""))
            if source_filter_lc and row_source.lower().strip() != source_filter_lc:
                continue

            row_video_path = row.get("video_path", "")
            if not row_video_path:
                continue

            w = parse_dimension(row.get("video_width", "0"))
            h = parse_dimension(row.get("video_height", "0"))
            if w <= 0 or h <= 0:
                continue

            key = (w, h)
            if key in by_resolution:
                continue

            by_resolution[key] = VideoRecord(
                label=str(row.get("label", "")),
                source=row_source,
                video_path=Path(row_video_path),
                video_width=w,
                video_height=h,
            )

    records = sorted(by_resolution.values(), key=lambda r: (r.video_width * r.video_height, r.video_width, r.video_height))
    if max_videos > 0:
        records = records[:max_videos]
    return records


def process_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    cleaned = cv2.medianBlur(mask, 5)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def find_main_motion(mask: np.ndarray) -> tuple[Optional[tuple[int, int, int, int]], Optional[tuple[int, int]], float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    best_center = None
    best_area = 0.0

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if area > best_area:
            best_area = area
            best_bbox = (x, y, w, h)
            best_center = (x + w // 2, y + h // 2)

    return best_bbox, best_center, best_area


def draw_trajectory(frame: np.ndarray, points: deque[tuple[int, int]]) -> None:
    pts = list(points)
    for i in range(1, len(pts)):
        p1 = pts[i - 1]
        p2 = pts[i]
        thickness = max(1, int(4 * (i / max(1, len(pts)))))
        cv2.line(frame, p1, p2, (0, 255, 255), thickness)


def run_demo(record: VideoRecord, current_index: Optional[int] = None, total_count: Optional[int] = None) -> bool:
    if not record.video_path.exists():
        raise FileNotFoundError(
            f"Plik wideo nie istnieje: {record.video_path}\n"
            "Sprawdz, czy sciezki w CSV sa poprawne na tym komputerze."
        )

    cap = cv2.VideoCapture(str(record.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie udalo sie otworzyc wideo: {record.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )

    trajectory: deque[tuple[int, int]] = deque(maxlen=TRAJECTORY_LENGTH)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    while True:
        loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        fg_mask = bg_subtractor.apply(frame)
        fg_mask = process_mask(fg_mask)

        bbox, center, area = find_main_motion(fg_mask)
        if center is not None:
            trajectory.append(center)

        overlay = frame.copy()

        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(overlay, center, 4, (0, 255, 255), -1)

        draw_trajectory(overlay, trajectory)

        # Prawy panel: maska ruchu
        mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        mask_bgr[:, :, 1] = np.maximum(mask_bgr[:, :, 1], mask_bgr[:, :, 0])

        panel = np.hstack((overlay, mask_bgr))

        order = ""
        if current_index is not None and total_count is not None:
            order = f"[{current_index}/{total_count}] "

        title = f"{order}label={record.label} | source={record.source} | video={record.video_path.name}"
        stats = f"motion_area={area:.0f} | traj_len={len(trajectory)}"
        hint = "q/ESC = wyjscie, n = nastepne"

        cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
        cv2.putText(panel, stats, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        cv2.putText(panel, hint, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        panel_display = cv2.resize(panel, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW_NAME, panel_display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            cap.release()
            cv2.destroyAllWindows()
            return True
        if key == ord("n"):
            break

        elapsed = time.perf_counter() - loop_start
        sleep_time = max(0.0, frame_interval - elapsed) + max(0.0, EXTRA_SLEEP_SECONDS)
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()
    return False


def run_resolution_sweep() -> None:
    records = load_records_distinct_resolutions(CSV_PATH, SWEEP_SOURCE_FILTER, SWEEP_MAX_VIDEOS)
    if not records:
        raise LookupError("Nie znaleziono rekordow do trybu resolution_sweep.")

    print(f"Tryb resolution_sweep: {len(records)} nagran")
    for i, rec in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] {rec.video_path.name} | {rec.video_width}x{rec.video_height} | {rec.label} | {rec.source}")
        should_quit = run_demo(rec, i, len(records))
        if should_quit:
            break


def main() -> None:
    if PLAY_MODE == "single":
        record = load_video_record(CSV_PATH, VIDEO_ID, SOURCE_FILTER)
        print(f"Uruchamiam BG Subtraction demo dla: {record.video_path}")
        print(f"label={record.label}, source={record.source}, src_res={record.video_width}x{record.video_height}")
        run_demo(record)
        return

    if PLAY_MODE == "resolution_sweep":
        run_resolution_sweep()
        return

    raise ValueError("PLAY_MODE musi byc 'single' albo 'resolution_sweep'.")


if __name__ == "__main__":
    main()
