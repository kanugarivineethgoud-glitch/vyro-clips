
import json
import logging
import sqlite3
import subprocess
import time
import uuid

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# -------------------------
# APP CONFIGURATION
# -------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vyro")

app = FastAPI(title="VYRO CLIPS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parent / "storage"
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "clips"
DATABASE = BASE / "jobs.sqlite3"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 250 * 1024 * 1024

executor = ThreadPoolExecutor(max_workers=1)
whisper_model = None


# -------------------------
# DATABASE
# -------------------------

def get_db():
    connection = sqlite3.connect(DATABASE, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                clips TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Safe migration for existing VYRO databases.
        columns = {
            row["name"]
            for row in db.execute(
                "PRAGMA table_info(jobs)"
            ).fetchall()
        }

        if "stage" not in columns:
            db.execute(
                "ALTER TABLE jobs ADD COLUMN stage TEXT DEFAULT 'queued'"
            )

        if "progress" not in columns:
            db.execute(
                "ALTER TABLE jobs ADD COLUMN progress INTEGER DEFAULT 0"
            )

        if "timings" not in columns:
            db.execute(
                "ALTER TABLE jobs ADD COLUMN timings TEXT DEFAULT '{}'"
            )


def create_job(job_id):
    with get_db() as db:
        db.execute(
            """
            INSERT INTO jobs (job_id, status, stage, progress)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, "queued", "Waiting", 0)
        )


def update_job(
    job_id,
    status=None,
    clips=None,
    error=None,
    stage=None,
    progress=None,
    timings=None
):
    fields = []
    values = []

    updates = {
        "status": status,
        "clips": json.dumps(clips) if clips is not None else None,
        "error": str(error)[:2000] if error is not None else None,
        "stage": stage,
        "progress": progress,
        "timings": json.dumps(timings) if timings is not None else None
    }

    for column, value in updates.items():
        if value is not None:
            fields.append(f"{column} = ?")
            values.append(value)

    if not fields:
        return

    values.append(job_id)

    with get_db() as db:
        db.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?",
            values
        )


def get_job(job_id):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,)
        ).fetchone()


init_db()


@app.on_event("startup")
def recover_interrupted_jobs():
    with get_db() as db:
        db.execute("""
            UPDATE jobs
            SET status = 'failed',
                stage = 'Interrupted',
                error = 'Processing was interrupted by a server restart. Please upload again.'
            WHERE status IN ('queued', 'analysing')
        """)


# -------------------------
# AI MODEL
# -------------------------

def get_whisper():
    global whisper_model

    if whisper_model is None:
        logger.info("Loading Whisper tiny model")

        whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            num_workers=1
        )

        logger.info("Whisper model loaded")

    return whisper_model


# -------------------------
# VIDEO UTILITIES
# -------------------------

def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60
    )

    return float(result.stdout.strip())


def detect_scenes(video_path):
    video = open_video(str(video_path))
    manager = SceneManager()

    manager.add_detector(
        ContentDetector(threshold=27.0)
    )

    manager.detect_scenes(video)
    return manager.get_scene_list()


# -------------------------
# CLIP ANALYSIS
# -------------------------

def analyse_video(job_id, video_path):
    started = time.perf_counter()

    timings = {
        "model_load_seconds": 0,
        "transcription_seconds": 0,
        "selection_seconds": 0,
        "export_seconds": 0
    }

    try:
        update_job(
            job_id,
            status="analysing",
            stage="Checking video",
            progress=5
        )

        duration = get_video_duration(video_path)

        if duration <= 0:
            raise Exception("Invalid video duration.")

        # Load or reuse the AI model.
        model_start = time.perf_counter()
        model = get_whisper()

        timings["model_load_seconds"] = round(
            time.perf_counter() - model_start, 2
        )

        # Faster transcription settings.
        update_job(
            job_id,
            stage="AI speech analysis",
            progress=15,
            timings=timings
        )

        transcription_start = time.perf_counter()

        segments, _ = model.transcribe(
            str(video_path),
            beam_size=1,
            best_of=1,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=True
        )

        speech = []

        for segment in segments:
            speech.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip()
            })

        timings["transcription_seconds"] = round(
            time.perf_counter() - transcription_start, 2
        )

        logger.info(
            "Job %s transcription took %.2f seconds",
            job_id,
            timings["transcription_seconds"]
        )

        # Find possible highlights.
        update_job(
            job_id,
            stage="Finding highlights",
            progress=55,
            timings=timings
        )

        selection_start = time.perf_counter()

        candidates = []

        keywords = [
            "wow", "amazing", "unbelievable",
            "important", "listen", "finally",
            "incredible", "clutch", "winner",
            "crazy", "no way", "let's go"
        ]

        for item in speech:
            words = item["text"].lower()
            score = 1

            for keyword in keywords:
                if keyword in words:
                    score += 2

            start = max(0, item["start"] - 8)
            end = min(duration, item["end"] + 12)

            if end > start:
                candidates.append({
                    "start": start,
                    "end": end,
                    "score": score,
                    "transcript": item["text"]
                })

        # Only use scene detection for longer videos
        # when speech-based highlights are unavailable.
        if not candidates and duration >= 120:
            try:
                scenes = detect_scenes(video_path)

                for scene_start, scene_end in scenes:
                    start = max(
                        0, scene_start.get_seconds()
                    )

                    end = min(
                        duration, scene_end.get_seconds()
                    )

                    if end > start:
                        candidates.append({
                            "start": start,
                            "end": end,
                            "score": 1,
                            "transcript": ""
                        })

            except Exception:
                logger.exception(
                    "Scene detection failed for job %s",
                    job_id
                )

        # Fallback for short videos without speech.
        if not candidates:
            candidates.append({
                "start": 0,
                "end": min(duration, 30),
                "score": 1,
                "transcript": ""
            })

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        selected = []

        for candidate in candidates:
            overlaps = any(
                candidate["start"] < clip["end"]
                and candidate["end"] > clip["start"]
                for clip in selected
            )

            if not overlaps:
                selected.append(candidate)

            if len(selected) >= 5:
                break

        selected.sort(
            key=lambda item: item["start"]
        )

        timings["selection_seconds"] = round(
            time.perf_counter() - selection_start, 2
        )

        if not selected:
            raise Exception("No useful highlights found.")

        # Export clips.
        update_job(
            job_id,
            stage="Creating clips",
            progress=65,
            timings=timings
        )

        export_start = time.perf_counter()
        results = []

        for index, clip in enumerate(selected, start=1):
            output = OUTPUTS / f"{job_id}_clip_{index}.mp4"

            clip_duration = clip["end"] - clip["start"]

            if clip_duration <= 0:
                continue

            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-ss", str(clip["start"]),
                    "-i", str(video_path),
                    "-t", str(clip_duration),
                    "-threads", "1",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    str(output)
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600
            )

            if not output.is_file() or output.stat().st_size == 0:
                raise Exception(
                    f"Clip {index} export produced an empty file."
                )

            results.append({
                "clip": index,
                "start": round(clip["start"], 2),
                "end": round(clip["end"], 2),
                "score": clip["score"],
                "transcript": clip["transcript"],
                "download": f"/clips/{output.name}"
            })

            progress = 65 + int(
                (index / len(selected)) * 30
            )

            update_job(
                job_id,
                stage=f"Exporting clip {index} of {len(selected)}",
                progress=progress
            )

        timings["export_seconds"] = round(
            time.perf_counter() - export_start, 2
        )

        if not results:
            raise Exception("No clips could be generated.")

        timings["total_seconds"] = round(
            time.perf_counter() - started, 2
        )

        logger.info(
            "Job %s completed. Timings: %s",
            job_id,
            timings
        )

        update_job(
            job_id,
            status="completed",
            stage="Completed",
            progress=100,
            clips=results,
            timings=timings
        )

    except Exception as error:
        logger.exception(
            "Processing failed for job %s",
            job_id
        )

        update_job(
            job_id,
            status="failed",
            stage="Failed",
            error=str(error)
        )


# -------------------------
# API
# -------------------------

@app.get("/")
def home():
    return {
        "message": "VYRO CLIPS API is running",
        "version": "2.1"
    }


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):

    filename = video.filename or ""

    if not filename.lower().endswith(
        (".mp4", ".mov", ".mkv", ".webm")
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    job_id = str(uuid.uuid4())
    extension = Path(filename).suffix.lower()
    video_path = UPLOADS / f"{job_id}{extension}"

    total_size = 0

    try:
        with video_path.open("wb") as buffer:
            while True:
                chunk = await video.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 250 MB upload limit"
                    )

                buffer.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty"
            )

        create_job(job_id)

        executor.submit(
            analyse_video,
            job_id,
            video_path
        )

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Video uploaded. Analysis started."
        }

    except HTTPException:
        video_path.unlink(missing_ok=True)
        raise

    except Exception as error:
        video_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(error)}"
        )

    finally:
        await video.close()


@app.get("/status/{job_id}")
def job_status(job_id: str):

    job = get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    response = {
        "status": job["status"],
        "clips": json.loads(job["clips"]),
        "stage": job["stage"],
        "progress": job["progress"],
        "timings": json.loads(job["timings"] or "{}")
    }

    if job["error"]:
        response["error"] = job["error"]

    return response


@app.get("/clips/{filename}")
def download_clip(filename: str):

    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename"
        )

    path = OUTPUTS / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Clip not found"
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename
    )
