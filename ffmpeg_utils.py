import json
import os
import shutil
import subprocess
from typing import Iterable

import cv2


def ffmpeg_available() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False, timeout=2)
        return result.returncode == 0
    except Exception:
        return False


def concat_mp4(paths: Iterable[str], out_path: str) -> bool:
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return False
    if not ffmpeg_available():
        return _concat_mp4_opencv(paths, out_path)
    list_path = f"{out_path}.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as handle:
            for path in paths:
                handle.write(f"file '{path}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=300)
        if result.returncode == 0:
            return True
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=300)
        return result.returncode == 0
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def mux_wav_into_mp4(video_path: str, wav_path: str, out_path: str) -> bool:
    if not ffmpeg_available():
        try:
            shutil.copyfile(video_path, out_path)
            return True
        except OSError:
            return False
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        wav_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=False, timeout=300)
    return result.returncode == 0


def concat_and_mux(video_paths: Iterable[str], wav_paths: Iterable[str], fps: int, out_path: str) -> bool:
    video_paths = [p for p in video_paths if p and os.path.exists(p)]
    wav_paths = [p for p in wav_paths if p and os.path.exists(p)]
    if not video_paths:
        return False
    if not ffmpeg_available():
        if not _concat_mp4_opencv(video_paths, out_path):
            return False
        if wav_paths:
            return mux_wav_into_mp4(out_path, wav_paths[0], out_path)
        return True
    list_path = f"{out_path}.txt"
    audio_path = None
    try:
        with open(list_path, "w", encoding="utf-8") as handle:
            for path in video_paths:
                handle.write(f"file '{path}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
        ]
        if wav_paths:
            audio_path = _concat_wav_ffmpeg(wav_paths, f"{out_path}.wav")
            if audio_path:
                cmd.extend(["-i", audio_path])
        cmd.extend(["-r", str(float(fps)), "-c:v", "libx264", "-pix_fmt", "yuv420p"])
        if audio_path:
            cmd.extend(["-c:a", "aac", "-shortest"])
        else:
            cmd.extend(["-an"])
        cmd.extend(["-movflags", "+faststart", out_path])
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=300)
        return result.returncode == 0
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass


def _concat_wav_ffmpeg(paths: list[str], out_path: str) -> str | None:
    list_path = f"{out_path}.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as handle:
            for path in paths:
                handle.write(f"file '{path}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c:a",
            "pcm_s16le",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=300)
        if result.returncode == 0:
            return out_path
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass
    return None


def _concat_mp4_opencv(paths: list[str], out_path: str) -> bool:
    if not paths:
        return False
    cap0 = cv2.VideoCapture(paths[0])
    if not cap0.isOpened():
        return False
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap0.get(cv2.CAP_PROP_FPS) or 24
    cap0.release()
    if width <= 0 or height <= 0:
        return False
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False
    try:
        for path in paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
            finally:
                cap.release()
        return True
    finally:
        writer.release()
