
import os
import uuid
import shutil
import subprocess
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

app = FastAPI(title="VYRO CLIPS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path("storage")
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "clips"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

jobs = {}
whisper_model = None


def get_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
    return whisper_model


def detect_scenes(video_path):
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=27.0))
    manager.detect_scenes(video)
    return manager.get_scene_list()


def analyse_video(job_id, video_path):
    try:
        jobs[job_id]["status"] = "analysing"

        model = get_whisper()
        segments, _ = model.transcribe(str(video_path))

        speech = [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip()
            }
            for s in segments
        ]

        scenes = detect_scenes(video_path)

        duration = 0
        if speech:
            duration = max(item["end"] for item in speech)

        candidates = []

        # Create candidate windows around speech segments.
        for item in speech:
            text = item["text"]
            words = text.lower()

            score = 1
            keywords = [
                "wow", "amazing", "unbelievable",
                "important", "listen", "finally",
                "incredible", "clutch", "winner",
                "crazy", "no way", "let's go"
            ]

            for keyword in keywords:
                if keyword in words:
                    score += 2

            start = max(0, item["start"] - 8)
            end = item["end"] + 12

            candidates.append({
                "start": start,
                "end": end,
                "score": score,
                "transcript": text
            })

        # If speech is unavailable, use scene changes as candidates.
        if not candidates:
            for scene_start, scene_end in scenes:
                candidates.append({
                    "start": max(0, scene_start.get_seconds()),
                    "end": scene_end.get_seconds(),
                    "score": 1,
                    "transcript": ""
                })

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # Select up to five non-overlapping candidate clips.
        selected = []
        for candidate in candidates:
            if candidate["end"] <= candidate["start"]:
                continue

            overlaps = any(
                candidate["start"] < clip["end"]
                and candidate["end"] > clip["start"]
                for clip in selected
            )

            if not overlaps:
                selected.append(candidate)

            if len(selected) >= 5:
                break

        selected.sort(key=lambda item: item["start"])

        results = []
        for index, clip in enumerate(selected, start=1):
            output = OUTPUTS / f"{job_id}_clip_{index}.mp4"

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(clip["start"]),
                    "-i", str(video_path),
                    "-t", str(clip["end"] - clip["start"]),
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    str(output)
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            results.append({
                "clip": index,
                "start": round(clip["start"], 2),
                "end": round(clip["end"], 2),
                "score": clip["score"],
                "transcript": clip["transcript"],
                "download": f"/clips/{output.name}"
            })

        jobs[job_id].update({
            "status": "completed",
            "clips": results
        })

    except Exception as error:
        jobs[job_id].update({
            "status": "failed",
            "error": str(error)
        })


@app.get("/")
def home():
    return {"message": "VYRO CLIPS API is running"}


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    if not video.filename.lower().endswith(
        (".mp4", ".mov", ".mkv", ".webm")
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    job_id = str(uuid.uuid4())
    video_path = UPLOADS / f"{job_id}.mp4"

    with video_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    jobs[job_id] = {
        "status": "queued",
        "clips": []
    }

    Thread(
        target=analyse_video,
        args=(job_id, video_path),
        daemon=True
    ).start()

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video uploaded. Analysis started."
    }


@app.get("/status/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/clips/{filename}")
def download_clip(filename: str):
    from fastapi.responses import FileResponse

    path = OUTPUTS / filename
    if not path.exists() or path.parent != OUTPUTS:
        raise HTTPException(status_code=404, detail="Clip not found")

    return FileResponse(path, media_type="video/mp4")
