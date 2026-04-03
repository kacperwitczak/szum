"""Demo: wykrywanie zmian polozen dloni (dense flow + Kalman).

Pipeline:
1) Dense optical flow (Farneback) miedzy klatkami.
2) Maska ruchu z magnitudy flow.
3) Wybor top-2 najbardziej dynamicznych komponentow (zwykle dlonie).
4) Stabilizacja pozycji przez filtr Kalmana (left/right hand).
5) Overlay trajektorii oraz predkosci ruchu.

Wymagania:
    pip install opencv-python numpy

Uruchomienie:
    python demo_denseflow_hands_kalman.py
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
WINDOW_NAME = "DenseFlow Hands + Kalman"
DISPLAY_SIZE = (1280, 760)
EXTRA_SLEEP_SECONDS = 0.02
SWEEP_SOURCE_FILTER: Optional[str] = None
SWEEP_MAX_VIDEOS = 8

# Dense flow
FLOW_PYRSCALE = 0.5
FLOW_LEVELS = 4
FLOW_WINSIZE = 23
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2
FLOW_MAG_THRESHOLD = 1.15

# Hands selection
HAND_COMPONENTS = 2
HAND_MIN_AREA = 140
HAND_UPPER_PORTION = 0.90
USE_SKIN_FILTER = True

# Kalman / tracking
MAX_ASSOCIATION_DISTANCE = 180.0
TRACK_TRAIL_LEN = 40


@dataclass
class VideoRecord:
    label: str
    source: str
    video_path: Path
    video_width: int
    video_height: int


@dataclass
class Detection:
    cx: float
    cy: float
    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass
class HandTrack:
    name: str
    kalman: cv2.KalmanFilter
    has_measurement: bool
    pos: Optional[tuple[float, float]]
    prev_pos: Optional[tuple[float, float]]
    speed_px_s: float
    trail: deque[tuple[int, int]]
    miss_count: int


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


def create_kalman_filter(x: float, y: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)

    # state: [x, y, vx, vy]
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 8e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
    return kf


def compute_skin_mask(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return skin


def select_hand_detections(frame: np.ndarray, mag: np.ndarray, motion_bin: np.ndarray) -> list[Detection]:
    h, w = motion_bin.shape

    upper = np.zeros((h, w), dtype=np.uint8)
    upper[: int(h * HAND_UPPER_PORTION), :] = 255
    mask = cv2.bitwise_and(motion_bin, upper)

    if USE_SKIN_FILTER:
        mask = cv2.bitwise_and(mask, compute_skin_mask(frame))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    dets: list[Detection] = []
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < HAND_MIN_AREA:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        ww = int(stats[label_id, cv2.CC_STAT_WIDTH])
        hh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        comp = labels == label_id
        mean_mag = float(np.mean(mag[comp]))
        upper_bias = 1.0 + 0.8 * (1.0 - (y / max(1.0, float(h))))
        score = area * mean_mag * upper_bias

        cx = x + ww * 0.5
        cy = y + hh * 0.5
        dets.append(Detection(cx=float(cx), cy=float(cy), x=x, y=y, w=ww, h=hh, score=score))

    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[: max(1, HAND_COMPONENTS)]


def assign_detections(left: HandTrack, right: HandTrack, dets: list[Detection], width: int) -> dict[str, Optional[Detection]]:
    assignment: dict[str, Optional[Detection]] = {"left": None, "right": None}
    if not dets:
        return assignment

    if len(dets) >= 2:
        dets_sorted = sorted(dets[:2], key=lambda d: d.cx)
        assignment["left"] = dets_sorted[0]
        assignment["right"] = dets_sorted[1]
        return assignment

    det = dets[0]
    mid = width * 0.5

    left_ref = left.pos[0] if left.pos is not None else mid * 0.5
    right_ref = right.pos[0] if right.pos is not None else mid * 1.5

    dist_left = abs(det.cx - left_ref)
    dist_right = abs(det.cx - right_ref)

    if min(dist_left, dist_right) > MAX_ASSOCIATION_DISTANCE and det.cx >= mid:
        assignment["right"] = det
    elif min(dist_left, dist_right) > MAX_ASSOCIATION_DISTANCE and det.cx < mid:
        assignment["left"] = det
    elif dist_left <= dist_right:
        assignment["left"] = det
    else:
        assignment["right"] = det

    return assignment


def update_track(track: HandTrack, det: Optional[Detection], fps: float) -> None:
    pred = track.kalman.predict()
    pred_pos = (float(pred[0, 0]), float(pred[1, 0]))

    if det is not None:
        meas = np.array([[det.cx], [det.cy]], dtype=np.float32)
        corrected = track.kalman.correct(meas)
        pos = (float(corrected[0, 0]), float(corrected[1, 0]))
        track.has_measurement = True
        track.miss_count = 0
    else:
        pos = pred_pos
        track.has_measurement = False
        track.miss_count += 1

    if track.pos is not None:
        dx = pos[0] - track.pos[0]
        dy = pos[1] - track.pos[1]
        track.speed_px_s = float(np.hypot(dx, dy) * fps)

    track.prev_pos = track.pos
    track.pos = pos
    track.trail.append((int(pos[0]), int(pos[1])))


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

    h, w = prev_gray.shape
    left_track = HandTrack(
        name="left",
        kalman=create_kalman_filter(w * 0.35, h * 0.45),
        has_measurement=False,
        pos=None,
        prev_pos=None,
        speed_px_s=0.0,
        trail=deque(maxlen=TRACK_TRAIL_LEN),
        miss_count=0,
    )
    right_track = HandTrack(
        name="right",
        kalman=create_kalman_filter(w * 0.65, h * 0.45),
        has_measurement=False,
        pos=None,
        prev_pos=None,
        speed_px_s=0.0,
        trail=deque(maxlen=TRACK_TRAIL_LEN),
        miss_count=0,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    while True:
        loop_start = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            FLOW_PYRSCALE,
            FLOW_LEVELS,
            FLOW_WINSIZE,
            FLOW_ITERATIONS,
            FLOW_POLY_N,
            FLOW_POLY_SIGMA,
            0,
        )
        fx = flow[..., 0]
        fy = flow[..., 1]
        mag, _ang = cv2.cartToPolar(fx, fy)

        motion_bin = (mag > FLOW_MAG_THRESHOLD).astype(np.uint8) * 255
        motion_bin = cv2.morphologyEx(motion_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        motion_bin = cv2.morphologyEx(motion_bin, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        detections = select_hand_detections(frame, mag, motion_bin)
        assignment = assign_detections(left_track, right_track, detections, w)

        update_track(left_track, assignment["left"], fps)
        update_track(right_track, assignment["right"], fps)

        overlay = frame.copy()
        hand_mask = np.zeros((h, w), dtype=np.uint8)

        for det in detections:
            cv2.rectangle(overlay, (det.x, det.y), (det.x + det.w, det.y + det.h), (0, 180, 255), 2)
            hand_mask[det.y : det.y + det.h, det.x : det.x + det.w] = 255

        frame_hands = cv2.bitwise_and(overlay, overlay, mask=hand_mask)

        for track, color in ((left_track, (0, 255, 0)), (right_track, (255, 180, 0))):
            if track.pos is None:
                continue

            px, py = int(track.pos[0]), int(track.pos[1])
            cv2.circle(frame_hands, (px, py), 6, color, -1)

            pts = np.array(list(track.trail), dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(frame_hands, [pts], False, color, 2, cv2.LINE_AA)

            speed_text = f"{track.name}: {track.speed_px_s:.1f}px/s"
            y_pos = 110 if track.name == "left" else 136
            cv2.putText(frame_hands, speed_text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        mag_u8 = np.clip((mag / max(1e-6, float(np.percentile(mag, 98)))) * 255.0, 0, 255).astype(np.uint8)
        flow_heat = cv2.applyColorMap(mag_u8, cv2.COLORMAP_TURBO)

        motion_bgr = cv2.cvtColor(motion_bin, cv2.COLOR_GRAY2BGR)

        vx = np.mean(fx[mag > FLOW_MAG_THRESHOLD]) if np.any(mag > FLOW_MAG_THRESHOLD) else 0.0
        vy = np.mean(fy[mag > FLOW_MAG_THRESHOLD]) if np.any(mag > FLOW_MAG_THRESHOLD) else 0.0
        arrows = frame.copy()
        step = 28
        for y0 in range(step // 2, h, step):
            for x0 in range(step // 2, w, step):
                dx = fx[y0, x0]
                dy = fy[y0, x0]
                if dx * dx + dy * dy < FLOW_MAG_THRESHOLD * FLOW_MAG_THRESHOLD:
                    continue
                end = (int(x0 + 3.0 * dx), int(y0 + 3.0 * dy))
                cv2.arrowedLine(arrows, (x0, y0), end, (0, 255, 255), 1, tipLength=0.25)

        top = np.hstack((frame_hands, flow_heat))
        bottom = np.hstack((motion_bgr, arrows))
        panel = np.vstack((top, bottom))

        active_ratio = float(np.mean(motion_bin > 0))
        order = ""
        if current_index is not None and total_count is not None:
            order = f"[{current_index}/{total_count}] "

        title = f"{order}{record.video_path.name} | label={record.label} | src={record.source}"
        stats = (
            f"active_motion={active_ratio:.2%} | detections={len(detections)} "
            f"| avg_v=({vx:.2f},{vy:.2f})"
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
        print(f"Uruchamiam DenseFlow+Kalman demo dla: {record.video_path}")
        print(f"label={record.label}, source={record.source}, src_res={record.video_width}x{record.video_height}")
        run_demo(record)
        return

    if PLAY_MODE == "resolution_sweep":
        run_resolution_sweep()
        return

    raise ValueError("PLAY_MODE musi byc 'single' albo 'resolution_sweep'.")


if __name__ == "__main__":
    main()
