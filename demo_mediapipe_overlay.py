"""Demo: wybór wideo z CSV i overlay punktów MediaPipe.

Wymagania:
    pip install opencv-python mediapipe

Uruchomienie:
    python demo_mediapipe_overlay.py
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Optional

import cv2
import mediapipe as mp


# =============================
# Konfiguracja demo (edytuj tutaj)
# =============================
CSV_PATH = Path("merged_datasets/universal_metadata_has_video_no_anomalies.csv")
VIDEO_ID = "31220"  # ID pliku wideo, np. 07069 -> ...\07069.mp4
SOURCE_FILTER: Optional[str] = "signingsavvy"  # np. "signschool" albo None
PLAY_MODE = "resolution_sweep"  # "single" albo "resolution_sweep"
WINDOW_NAME = "MediaPipe demo"
DRAW_FACE = False  # FaceMesh jest kosztowny; domyślnie wyłączone dla płynności
EXTRA_SLEEP_SECONDS = 0.2  # np. 0.02 spowolni każdą klatkę o dodatkowe 20 ms
DISPLAY_SIZE = (960, 540)  # (szerokosc, wysokosc) stale dla wszystkich filmow
SWEEP_SOURCE_FILTER: Optional[str] = None  # np. "signingsavvy" lub None
SWEEP_MAX_VIDEOS = 20  # ile różnych rozdzielczości odtworzyć kolejno
MODEL_COMPLEXITY = 2  # 2 = stabilniej, ale wolniej
MIN_DETECTION_CONFIDENCE = 0.60
MIN_TRACKING_CONFIDENCE = 0.60
USE_TEMPORAL_EMA = True
EMA_ALPHA = 0.6  # mniejsze = mocniejsze wygładzenie, większe = szybsza reakcja


@dataclass
class VideoRecord:
    label: str
    source: str
    video_path: Path
    video_width: int
    video_height: int


def ema_smooth_landmarks(current, previous, alpha: float):
    """Wygładza landmarki z bieżącej klatki względem poprzedniej (EMA)."""
    if current is None:
        return None, None

    if previous is None:
        return current, current

    count = min(len(current.landmark), len(previous.landmark))
    for i in range(count):
        curr_lm = current.landmark[i]
        prev_lm = previous.landmark[i]

        curr_lm.x = alpha * curr_lm.x + (1.0 - alpha) * prev_lm.x
        curr_lm.y = alpha * curr_lm.y + (1.0 - alpha) * prev_lm.y
        curr_lm.z = alpha * curr_lm.z + (1.0 - alpha) * prev_lm.z

        # visibility występuje np. dla pose
        if hasattr(curr_lm, "visibility") and hasattr(prev_lm, "visibility"):
            curr_lm.visibility = alpha * curr_lm.visibility + (1.0 - alpha) * prev_lm.visibility

    return current, current


def normalize_video_id(value: str) -> str:
    """Normalizuje ID do porównania niezależnie od zer wiodących."""
    value = str(value).strip()
    if not value:
        raise ValueError("VIDEO_ID nie może być pusty.")
    if value.isdigit():
        return str(int(value))
    return value


def get_video_id_from_path(video_path: str) -> str:
    """Wyciąga ID z nazwy pliku, np. C:\\...\\07069.mp4 -> 7069."""
    stem = PureWindowsPath(video_path).stem
    return normalize_video_id(stem)


def parse_dimension(value: str) -> int:
    """Konwersja typu 288.0/288 do liczby całkowitej pikseli."""
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

            row_source = row.get("source", "")
            if source_filter_lc and row_source.lower().strip() != source_filter_lc:
                continue

            return VideoRecord(
                label=row.get("label", ""),
                source=row_source,
                video_path=Path(row_video_path),
                video_width=parse_dimension(row.get("video_width", "0")),
                video_height=parse_dimension(row.get("video_height", "0")),
            )

    hint = f" i źródła '{source_filter}'" if source_filter else ""
    raise LookupError(f"Nie znaleziono rekordu dla VIDEO_ID='{video_id}'{hint}.")


def load_records_distinct_resolutions(
    csv_path: Path,
    source_filter: Optional[str],
    max_videos: int,
) -> list[VideoRecord]:
    """Wczytuje rekordy z unikalnymi rozdzielczościami i sortuje je po liczbie pikseli."""
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
            "Sprawdź, czy ścieżki w CSV są poprawne na tym komputerze."
        )

    cap = cv2.VideoCapture(str(record.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie udało się otworzyć wideo: {record.video_path}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    prev_pose = None
    prev_left_hand = None
    prev_right_hand = None
    prev_face = None

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as holistic:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            if USE_TEMPORAL_EMA:
                results.pose_landmarks, prev_pose = ema_smooth_landmarks(results.pose_landmarks, prev_pose, EMA_ALPHA)
                results.left_hand_landmarks, prev_left_hand = ema_smooth_landmarks(results.left_hand_landmarks, prev_left_hand, EMA_ALPHA)
                results.right_hand_landmarks, prev_right_hand = ema_smooth_landmarks(results.right_hand_landmarks, prev_right_hand, EMA_ALPHA)
                if DRAW_FACE:
                    results.face_landmarks, prev_face = ema_smooth_landmarks(results.face_landmarks, prev_face, EMA_ALPHA)

            # Pose
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            # Lewa ręka
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                )

            # Prawa ręka
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                )

            # Opcjonalna twarz
            if DRAW_FACE and results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )

            order = ""
            if current_index is not None and total_count is not None:
                order = f"[{current_index}/{total_count}] "

            overlay = f"{order}label={record.label} | source={record.source} | video={record.video_path.name}"
            src_res = f"src_res={record.video_width}x{record.video_height}"
            cv2.putText(frame, overlay, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, src_res, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, "q/ESC = wyjscie, n = nastepne wideo", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"ema={USE_TEMPORAL_EMA} alpha={EMA_ALPHA}", (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            frame_display = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, frame_display)
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
        raise LookupError("Nie znaleziono rekordów do trybu resolution_sweep.")

    print(f"Tryb resolution_sweep: {len(records)} nagrań z różnymi rozdzielczościami")
    for i, rec in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] {rec.video_path.name} | {rec.video_width}x{rec.video_height} | {rec.label} | {rec.source}")
        should_quit = run_demo(rec, i, len(records))
        if should_quit:
            break


def main() -> None:
    if PLAY_MODE == "single":
        record = load_video_record(CSV_PATH, VIDEO_ID, SOURCE_FILTER)
        print(f"Uruchamiam demo dla: {record.video_path}")
        print(f"label={record.label}, source={record.source}")
        run_demo(record)
        return

    if PLAY_MODE == "resolution_sweep":
        run_resolution_sweep()
        return

    raise ValueError("PLAY_MODE musi być 'single' albo 'resolution_sweep'.")


if __name__ == "__main__":
    main()
