"""Demo alternatywne do MediaPipe: analiza ruchu (dense optical flow).

To podejscie nie wykrywa landmarkow ciala i dloni.
Zamiast tego modeluje ruch pikseli miedzy klatkami.

Wymagania:
    pip install opencv-python numpy

Uruchomienie:
    python demo_optical_flow_sign.py
"""

from __future__ import annotations

import csv
import time
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
WINDOW_NAME = "Optical Flow demo"
DISPLAY_SIZE = (960, 540)
EXTRA_SLEEP_SECONDS = 0.03  # lekkie spowolnienie: +30 ms na klatke
SWEEP_SOURCE_FILTER: Optional[str] = None
SWEEP_MAX_VIDEOS = 20

# Parametry optical flow / filtrowania szumu
FLOW_MAG_THRESHOLD = 1.2  # wyzszy prog = mniej szumu, mniej slabego ruchu
ARROW_STEP = 18  # siatka strzalek (piksele)
ARROW_MIN_MAG = 2.0
ARROW_SCALE = 2.5
BLUR_KERNEL = 5  # wygladzenie maski ruchu


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


def draw_flow_arrows(frame: np.ndarray, flow: np.ndarray, step: int, min_mag: float, scale: float) -> None:
    h, w = frame.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]
            mag = float(np.hypot(fx, fy))
            if mag < min_mag:
                continue

            x2 = int(x + fx * scale)
            y2 = int(y + fy * scale)
            cv2.arrowedLine(frame, (x, y), (x2, y2), (0, 255, 255), 1, tipLength=0.3)


def build_motion_overlay(frame: np.ndarray, mag: np.ndarray, threshold: float) -> np.ndarray:
    # Tworzy czytelna maske ruchu: czerwone obszary tam, gdzie ruch > prog.
    motion_mask = (mag > threshold).astype(np.uint8) * 255

    if BLUR_KERNEL > 1:
        motion_mask = cv2.GaussianBlur(motion_mask, (BLUR_KERNEL, BLUR_KERNEL), 0)

    heat = np.zeros_like(frame)
    heat[:, :, 2] = motion_mask  # kanal R

    blended = cv2.addWeighted(frame, 0.85, heat, 0.45, 0.0)
    return blended


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

    ok, prev_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Nie mozna odczytac pierwszej klatki.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    while True:
        loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        fx = flow[:, :, 0]
        fy = flow[:, :, 1]
        mag = np.sqrt(fx * fx + fy * fy)

        frame_overlay = build_motion_overlay(frame, mag, FLOW_MAG_THRESHOLD)
        draw_flow_arrows(frame_overlay, flow, ARROW_STEP, ARROW_MIN_MAG, ARROW_SCALE)

        mean_mag = float(np.mean(mag))
        p95_mag = float(np.percentile(mag, 95))
        active_ratio = float(np.mean(mag > FLOW_MAG_THRESHOLD))

        title = f"label={record.label} | source={record.source} | video={record.video_path.name}"
        stats = f"mean={mean_mag:.3f} p95={p95_mag:.3f} active={active_ratio:.2%}"
        order = ""
        if current_index is not None and total_count is not None:
            order = f"[{current_index}/{total_count}] "

        hint = "q/ESC = wyjscie, n = nastepne"

        cv2.putText(frame_overlay, f"{order}{title}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
        cv2.putText(frame_overlay, stats, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        cv2.putText(frame_overlay, hint, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        frame_display = cv2.resize(frame_overlay, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW_NAME, frame_display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            cap.release()
            cv2.destroyAllWindows()
            return True
        if key == ord("n"):
            break

        prev_gray = gray

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
        print(f"Uruchamiam Optical Flow demo dla: {record.video_path}")
        print(f"label={record.label}, source={record.source}, src_res={record.video_width}x{record.video_height}")
        run_demo(record)
        return

    if PLAY_MODE == "resolution_sweep":
        run_resolution_sweep()
        return

    raise ValueError("PLAY_MODE musi byc 'single' albo 'resolution_sweep'.")


if __name__ == "__main__":
    main()
