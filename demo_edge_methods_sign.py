"""Demo: porownanie metod wykrywania krawedzi dla nagran znakow.

Panel 2x2:
- oryginal,
- Canny,
- Sobel magnitude,
- Laplacian.

Wymagania:
    pip install opencv-python numpy

Uruchomienie:
    python demo_edge_methods_sign.py
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
WINDOW_NAME = "Edge methods demo"
DISPLAY_SIZE = (1280, 760)
EXTRA_SLEEP_SECONDS = 0.02
SWEEP_SOURCE_FILTER: Optional[str] = None
SWEEP_MAX_VIDEOS = 8
HANDS_ONLY_MODE = True

# Parametry krawedzi
GAUSS_BLUR = 5
CANNY_LOW = 70
CANNY_HIGH = 160
SOBEL_KSIZE = 3
LAPLACIAN_KSIZE = 3
MOTION_DIFF_THRESHOLD = 16

# Parametry dynamicznej segmentacji dloni
HAND_TRAIL_DECAY = 0.90
HAND_DYN_THRESHOLD = 0.28  # 0..1
HAND_COMPONENTS = 2
HAND_MIN_AREA = 140
HAND_UPPER_PORTION = 0.88
USE_SKIN_FILTER = True


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
    return normalize_video_id(PureWindowsPath(video_path).stem)


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

            if get_video_id_from_path(row_video_path) != wanted_id:
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


def normalize_to_u8(src: np.ndarray) -> np.ndarray:
    src = np.abs(src)
    max_v = float(np.max(src))
    if max_v <= 1e-6:
        return np.zeros_like(src, dtype=np.uint8)
    return np.clip((src / max_v) * 255.0, 0, 255).astype(np.uint8)


def render_edges(gray: np.ndarray, prev_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blur = cv2.GaussianBlur(gray, (GAUSS_BLUR, GAUSS_BLUR), 0)

    canny = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE)
    sobel_mag = cv2.magnitude(gx, gy)
    sobel_u8 = normalize_to_u8(sobel_mag)

    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=LAPLACIAN_KSIZE)
    lap_u8 = normalize_to_u8(lap)

    # maska ruchu, by mierzyc aktywnosc na krawedziach, a nie szum tla
    diff = cv2.absdiff(gray, prev_gray)
    _, motion = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return canny, sobel_u8, lap_u8, motion


def compute_skin_mask(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Przyblizona segmentacja skory (dziala rozsadnie dla wielu warunkow, ale nie idealnie).
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return skin


def extract_dynamic_hand_mask(frame: np.ndarray, motion: np.ndarray, hand_trail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hand_trail *= HAND_TRAIL_DECAY
    hand_trail = np.maximum(hand_trail, (motion > 0).astype(np.float32))

    trail_u8 = np.clip(hand_trail * 255, 0, 255).astype(np.uint8)
    _, dyn_bin = cv2.threshold(trail_u8, int(HAND_DYN_THRESHOLD * 255), 255, cv2.THRESH_BINARY)

    h, w = motion.shape
    upper_mask = np.zeros((h, w), dtype=np.uint8)
    upper_mask[: int(h * HAND_UPPER_PORTION), :] = 255
    dyn_bin = cv2.bitwise_and(dyn_bin, upper_mask)

    if USE_SKIN_FILTER:
        dyn_bin = cv2.bitwise_and(dyn_bin, compute_skin_mask(frame))

    dyn_bin = cv2.morphologyEx(dyn_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dyn_bin = cv2.morphologyEx(dyn_bin, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dyn_bin, connectivity=8)

    scored: list[tuple[float, int]] = []
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < HAND_MIN_AREA:
            continue

        y = int(stats[label_id, cv2.CC_STAT_TOP])
        comp = labels == label_id
        mean_dyn = float(np.mean(hand_trail[comp]))
        upper_bias = 1.0 + 0.9 * (1.0 - (y / max(1.0, float(h))))
        score = mean_dyn * area * upper_bias
        scored.append((score, label_id))

    scored.sort(reverse=True)
    chosen = scored[: max(1, HAND_COMPONENTS)]

    hand_mask = np.zeros((h, w), dtype=np.uint8)
    for _score, label_id in chosen:
        hand_mask[labels == label_id] = 255

    hand_mask = cv2.dilate(hand_mask, np.ones((5, 5), np.uint8), iterations=1)
    return hand_mask, hand_trail


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
    fps = fps if fps and fps > 0 else 30.0
    frame_interval = 1.0 / fps

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Nie mozna odczytac pierwszej klatki")

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hand_trail = np.zeros(prev_gray.shape, dtype=np.float32)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    while True:
        loop_start = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny, sobel_u8, lap_u8, motion = render_edges(gray, prev_gray)

        if HANDS_ONLY_MODE:
            hand_mask, hand_trail = extract_dynamic_hand_mask(frame, motion, hand_trail)
            active_ratio = float(np.mean((canny > 0) & (hand_mask > 0)))
            hand_pixels = int(np.count_nonzero(hand_mask))

            frame_masked = cv2.bitwise_and(frame, frame, mask=hand_mask)
            canny_masked = cv2.bitwise_and(canny, canny, mask=hand_mask)
            sobel_masked = cv2.bitwise_and(sobel_u8, sobel_u8, mask=hand_mask)
            lap_masked = cv2.bitwise_and(lap_u8, lap_u8, mask=hand_mask)

            canny_bgr = cv2.cvtColor(canny_masked, cv2.COLOR_GRAY2BGR)
            sobel_bgr = cv2.cvtColor(sobel_masked, cv2.COLOR_GRAY2BGR)
            lap_bgr = cv2.cvtColor(lap_masked, cv2.COLOR_GRAY2BGR)

            # Obrys dloni dla czytelniejszego podgladu.
            contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_masked, contours, -1, (0, 255, 0), 2)
            top = np.hstack((frame_masked, canny_bgr))
            bottom = np.hstack((sobel_bgr, lap_bgr))
        else:
            active_ratio = float(np.mean((canny > 0) & (motion > 0)))
            hand_pixels = -1

            canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
            sobel_bgr = cv2.cvtColor(sobel_u8, cv2.COLOR_GRAY2BGR)
            lap_bgr = cv2.cvtColor(lap_u8, cv2.COLOR_GRAY2BGR)
            top = np.hstack((frame, canny_bgr))
            bottom = np.hstack((sobel_bgr, lap_bgr))

        panel = np.vstack((top, bottom))

        order = ""
        if current_index is not None and total_count is not None:
            order = f"[{current_index}/{total_count}] "

        title = f"{order}{record.video_path.name} | label={record.label} | src={record.source}"
        if HANDS_ONLY_MODE:
            stats = (
                f"active_edges_hands={active_ratio:.2%} | hand_px={hand_pixels} "
                f"| canny=({CANNY_LOW},{CANNY_HIGH}) | sobel_k={SOBEL_KSIZE} | lap_k={LAPLACIAN_KSIZE}"
            )
        else:
            stats = (
                f"active_edges={active_ratio:.2%} | canny=({CANNY_LOW},{CANNY_HIGH}) "
                f"| sobel_k={SOBEL_KSIZE} | lap_k={LAPLACIAN_KSIZE}"
            )
        hint = "q/ESC = wyjscie, n = nastepne"

        cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
        cv2.putText(panel, stats, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)
        cv2.putText(panel, hint, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        panel_display = cv2.resize(panel, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW_NAME, panel_display)

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
        print(f"Uruchamiam edge methods demo dla: {record.video_path}")
        print(f"label={record.label}, source={record.source}, src_res={record.video_width}x{record.video_height}")
        run_demo(record)
        return

    if PLAY_MODE == "resolution_sweep":
        run_resolution_sweep()
        return

    raise ValueError("PLAY_MODE musi byc 'single' albo 'resolution_sweep'.")


if __name__ == "__main__":
    main()
