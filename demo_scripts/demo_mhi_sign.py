"""Demo alternatywne: MEI + MHI (Motion Energy / Motion History Image).

MEI: gdzie byl ruch.
MHI: jak swiezy byl ruch (jasniejsze = nowszy).

Wymagania:
    pip install opencv-python numpy

Uruchomienie:
    python demo_mhi_sign.py
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
WINDOW_NAME = "MEI/MHI demo"
CONTROL_WINDOW = "MEI/MHI controls"
DISPLAY_SIZE = (1200, 700)
EXTRA_SLEEP_SECONDS = 0.02
SWEEP_SOURCE_FILTER: Optional[str] = None
SWEEP_MAX_VIDEOS = 30
USE_TRACKBARS = True
COMPARE_VARIANTS = True
VARIANT_GRID_COLUMNS = 2

# Parametry MHI
DIFF_THRESHOLD = 22
MHI_DURATION_SECONDS = 1.2
MEI_DECAY = 0.92  # 0.8-0.98: wyzsze = dluzsza "pamiec" energii ruchu
BLUR_KERNEL = 5

# Presety do automatycznego porownania na jednym ekranie.
VARIANT_PRESETS = [
    {
        "name": "v1_sensitive_short",
        "diff_threshold": 14,
        "mhi_duration": 0.70,
        "mei_decay": 0.84,
        "blur_kernel": 3,
    },
    {
        "name": "v2_balanced",
        "diff_threshold": 22,
        "mhi_duration": 1.20,
        "mei_decay": 0.92,
        "blur_kernel": 5,
    },
    {
        "name": "v3_stable_long",
        "diff_threshold": 30,
        "mhi_duration": 1.80,
        "mei_decay": 0.96,
        "blur_kernel": 7,
    },
    {
        "name": "v4_strict_clean",
        "diff_threshold": 38,
        "mhi_duration": 1.10,
        "mei_decay": 0.90,
        "blur_kernel": 9,
    },
]


def _noop(_: int) -> None:
    return


def create_control_panel() -> None:
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 480, 220)

    cv2.createTrackbar("diff_threshold", CONTROL_WINDOW, int(DIFF_THRESHOLD), 255, _noop)
    cv2.createTrackbar("mhi_duration_x100", CONTROL_WINDOW, int(MHI_DURATION_SECONDS * 100), 800, _noop)
    cv2.createTrackbar("mei_decay_x100", CONTROL_WINDOW, int(MEI_DECAY * 100), 100, _noop)

    # 0..15 mapujemy na nieparzyste 1..31
    initial_blur_index = max(0, min(15, (BLUR_KERNEL - 1) // 2))
    cv2.createTrackbar("blur_kernel_idx", CONTROL_WINDOW, initial_blur_index, 15, _noop)


def read_live_params() -> tuple[int, float, float, int]:
    diff_threshold = max(0, cv2.getTrackbarPos("diff_threshold", CONTROL_WINDOW))

    mhi_duration = cv2.getTrackbarPos("mhi_duration_x100", CONTROL_WINDOW) / 100.0
    mhi_duration = max(0.05, mhi_duration)

    mei_decay = cv2.getTrackbarPos("mei_decay_x100", CONTROL_WINDOW) / 100.0
    mei_decay = float(np.clip(mei_decay, 0.0, 1.0))

    blur_index = cv2.getTrackbarPos("blur_kernel_idx", CONTROL_WINDOW)
    blur_kernel = max(1, 2 * blur_index + 1)

    return diff_threshold, mhi_duration, mei_decay, blur_kernel


@dataclass
class VideoRecord:
    label: str
    source: str
    video_path: Path
    video_width: int
    video_height: int


@dataclass
class VariantConfig:
    name: str
    diff_threshold: int
    mhi_duration: float
    mei_decay: float
    blur_kernel: int


@dataclass
class VariantState:
    config: VariantConfig
    mhi: np.ndarray
    mei: np.ndarray


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


def normalize_blur_kernel(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def build_variant_configs() -> list[VariantConfig]:
    configs: list[VariantConfig] = []
    for item in VARIANT_PRESETS:
        configs.append(
            VariantConfig(
                name=str(item.get("name", "variant")),
                diff_threshold=max(0, int(item.get("diff_threshold", DIFF_THRESHOLD))),
                mhi_duration=max(0.05, float(item.get("mhi_duration", MHI_DURATION_SECONDS))),
                mei_decay=float(np.clip(float(item.get("mei_decay", MEI_DECAY)), 0.0, 1.0)),
                blur_kernel=normalize_blur_kernel(int(item.get("blur_kernel", BLUR_KERNEL))),
            )
        )
    return configs


def build_variant_states(height: int, width: int, configs: list[VariantConfig]) -> list[VariantState]:
    states: list[VariantState] = []
    for cfg in configs:
        states.append(
            VariantState(
                config=cfg,
                mhi=np.zeros((height, width), dtype=np.float32),
                mei=np.zeros((height, width), dtype=np.float32),
            )
        )
    return states


def make_grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    if not tiles:
        return np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8)

    columns = max(1, int(columns))
    tile_h, tile_w = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns

    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    padded = tiles + [blank] * (rows * columns - len(tiles))

    row_strips = []
    for r in range(rows):
        start = r * columns
        end = start + columns
        row_strips.append(np.hstack(padded[start:end]))
    return np.vstack(row_strips)


def render_variant_tile(
    frame: np.ndarray,
    diff: np.ndarray,
    timestamp: float,
    variant: VariantState,
) -> np.ndarray:
    cfg = variant.config
    kernel = np.ones((cfg.blur_kernel, cfg.blur_kernel), np.uint8)

    _, motion_mask = cv2.threshold(diff, cfg.diff_threshold, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.GaussianBlur(motion_mask, (cfg.blur_kernel, cfg.blur_kernel), 0)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    cv2.motempl.updateMotionHistory(motion_mask, variant.mhi, timestamp, cfg.mhi_duration)

    variant.mei *= cfg.mei_decay
    variant.mei = np.maximum(variant.mei, (motion_mask > 0).astype(np.float32))

    mhi_norm = np.clip((variant.mhi - (timestamp - cfg.mhi_duration)) / cfg.mhi_duration, 0.0, 1.0)
    mhi_vis = (mhi_norm * 255).astype(np.uint8)
    mhi_color = cv2.applyColorMap(mhi_vis, cv2.COLORMAP_TURBO)
    mask_bw = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    frame_overlay = frame.copy()
    active = motion_mask > 0
    frame_overlay[active] = (0.35 * frame_overlay[active] + 0.65 * np.array([0, 80, 255])).astype(np.uint8)

    tile = np.hstack((frame_overlay, mask_bw, mhi_color))
    active_ratio = float(np.mean(active))

    line1 = f"{cfg.name} | active={active_ratio:.2%}"
    line2 = (
        f"T={cfg.diff_threshold} dur={cfg.mhi_duration:.2f}s "
        f"decay={cfg.mei_decay:.2f} blur={cfg.blur_kernel}"
    )
    cv2.putText(tile, line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
    cv2.putText(tile, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return tile


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

    ok, prev_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Nie mozna odczytac pierwszej klatki")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    use_compare_variants = COMPARE_VARIANTS and len(VARIANT_PRESETS) > 0
    use_trackbars = USE_TRACKBARS and not use_compare_variants

    variants: list[VariantState] = []
    if use_compare_variants:
        variants = build_variant_states(h, w, build_variant_configs())

    if use_trackbars:
        create_control_panel()

    diff_threshold = int(DIFF_THRESHOLD)
    mhi_duration = float(MHI_DURATION_SECONDS)
    mei_decay = float(MEI_DECAY)
    blur_kernel = int(BLUR_KERNEL)
    mhi = np.zeros((h, w), dtype=np.float32)
    mei = np.zeros((h, w), dtype=np.float32)

    while True:
        loop_start = time.perf_counter()

        if use_trackbars:
            diff_threshold, mhi_duration, mei_decay, blur_kernel = read_live_params()

        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray, prev_gray)
        timestamp = time.time()
        order = ""
        if current_index is not None and total_count is not None:
            order = f"[{current_index}/{total_count}] "

        title = f"{order}{record.video_path.name} | label={record.label} | src={record.source}"
        if use_compare_variants:
            tiles = [render_variant_tile(frame, diff, timestamp, var) for var in variants]
            panel = make_grid(tiles, VARIANT_GRID_COLUMNS)
            hint = "q/ESC = wyjscie, n = nastepne | porownanie presetow"
            cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            cv2.putText(panel, hint, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        else:
            kernel = np.ones((blur_kernel, blur_kernel), np.uint8)
            _, motion_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.GaussianBlur(motion_mask, (blur_kernel, blur_kernel), 0)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

            cv2.motempl.updateMotionHistory(motion_mask, mhi, timestamp, mhi_duration)

            # MEI jako wygaszana energia ruchu
            mei *= mei_decay
            mei = np.maximum(mei, (motion_mask > 0).astype(np.float32))

            # Normalizacja MHI do podgladu
            mhi_norm = np.clip((mhi - (timestamp - mhi_duration)) / mhi_duration, 0.0, 1.0)
            mhi_vis = (mhi_norm * 255).astype(np.uint8)
            mei_vis = np.clip(mei * 255, 0, 255).astype(np.uint8)

            mhi_color = cv2.applyColorMap(mhi_vis, cv2.COLORMAP_TURBO)
            mei_color = cv2.applyColorMap(mei_vis, cv2.COLORMAP_INFERNO)
            mask_color = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

            # 2x2 panel: frame / mask / MEI / MHI
            top = np.hstack((frame, mask_color))
            bottom = np.hstack((mei_color, mhi_color))
            panel = np.vstack((top, bottom))

            active_ratio = float(np.mean(motion_mask > 0))
            stats = (
                f"active={active_ratio:.2%} | DIFF_T={diff_threshold} | MHI_dur={mhi_duration:.2f}s "
                f"| MEI_decay={mei_decay:.2f} | blur={blur_kernel}"
            )
            hint = "q/ESC = wyjscie, n = nastepne | strojenie: okno controls"

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
        print(f"Uruchamiam MEI/MHI demo dla: {record.video_path}")
        print(f"label={record.label}, source={record.source}, src_res={record.video_width}x{record.video_height}")
        run_demo(record)
        return

    if PLAY_MODE == "resolution_sweep":
        run_resolution_sweep()
        return

    raise ValueError("PLAY_MODE musi byc 'single' albo 'resolution_sweep'.")


if __name__ == "__main__":
    main()
