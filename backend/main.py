import os
import re
import json
import uuid
import time
import sqlite3
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------
# VYRO CLIPS — BACKEND
# ---------------------------------------------------------

VERSION = "4.1"

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
CLIP_DIR = STORAGE_DIR / "clips"
DB_PATH = STORAGE_DIR / "jobs.sqlite3"

for folder in (STORAGE_DIR, UPLOAD_DIR, CLIP_DIR):
    folder.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 250 * 1024 * 1024
MAX_CLIPS = 5
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vyro")

app = FastAPI(
    title="VYRO CLIPS API",
    version=VERSION,
    description="AI video highlight analysis and YouTube intelligence"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=1)

# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

def db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    with db_connection() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'queued',
                stage TEXT DEFAULT 'Queued',
                progress INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                video_name TEXT,
                source_mode TEXT DEFAULT 'upload',
                youtube_url TEXT,
                clips TEXT DEFAULT '[]',
                error TEXT
            )
        """)

        columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(jobs)").fetchall()
        }

        migrations = {
            "stage": "TEXT DEFAULT 'Queued'",
            "progress": "INTEGER DEFAULT 0",
            "updated_at": "TEXT",
            "video_name": "TEXT",
            "source_mode": "TEXT DEFAULT 'upload'",
            "youtube_url": "TEXT",
            "clips": "TEXT DEFAULT '[]'",
            "error": "TEXT",
        }

        for name, definition in migrations.items():
            if name not in columns:
                db.execute(
                    f"ALTER TABLE jobs ADD COLUMN {name} {definition}"
                )

        db.commit()


init_database()


def create_job(job_id, video_name, source_mode, youtube_url):
    now = datetime.now(timezone.utc).isoformat()

    with db_connection() as db:
        db.execute("""
            INSERT INTO jobs (
                job_id, status, stage, progress,
                created_at, updated_at, video_name,
                source_mode, youtube_url, clips
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id,
            "queued",
            "Waiting for processing",
            0,
            now,
            now,
            video_name,
            source_mode,
            youtube_url,
            "[]",
        ))
        db.commit()


def update_job(job_id, **fields):
    allowed = {
        "status",
        "stage",
        "progress",
        "clips",
        "error",
        "video_name",
        "source_mode",
        "youtube_url",
    }

    updates = {
        key: value
        for key, value in fields.items()
        if key in allowed
    }

    if not updates:
        return

    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    assignments = ", ".join(
        f"{key} = ?" for key in updates
    )

    values = list(updates.values())
    values.append(job_id)

    with db_connection() as db:
        db.execute(
            f"UPDATE jobs SET {assignments} WHERE job_id = ?",
            values
        )
        db.commit()


def get_job(job_id):
    with db_connection() as db:
        row = db.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,)
        ).fetchone()

    if not row:
        return None

    result = dict(row)

    try:
        result["clips"] = json.loads(result.get("clips") or "[]")
    except Exception:
        result["clips"] = []

    return result


# ---------------------------------------------------------
# YOUTUBE URL VALIDATION
# ---------------------------------------------------------

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtube-nocookie.com",
}


def extract_youtube_video_id(url):
    if not isinstance(url, str) or not url.strip():
        raise HTTPException(400, "Please enter a YouTube URL.")

    try:
        parsed = urlparse(url.strip())
    except Exception:
        raise HTTPException(400, "Invalid YouTube URL.")

    if parsed.scheme != "https":
        raise HTTPException(400, "Only HTTPS YouTube links are accepted.")

    host = (parsed.hostname or "").lower()

    if host not in YOUTUBE_HOSTS:
        raise HTTPException(400, "Please enter a valid YouTube URL.")

    video_id = None

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]

    elif parsed.path == "/watch":
        from urllib.parse import parse_qs
        video_id = parse_qs(parsed.query).get("v", [None])[0]

    else:
        parts = [part for part in parsed.path.split("/") if part]

        if len(parts) >= 2 and parts[0] in {
            "live", "shorts", "embed"
        }:
            video_id = parts[1]

    if not video_id or not re.fullmatch(
        r"[A-Za-z0-9_-]{11}", video_id
    ):
        raise HTTPException(400, "Could not identify the YouTube video.")

    return video_id


def canonical_youtube_url(url):
    video_id = extract_youtube_video_id(url)
    return f"https://www.youtube.com/watch?v={video_id}"


# ---------------------------------------------------------
# YOUTUBE DATA API
# ---------------------------------------------------------

def youtube_api_get(endpoint, params):
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            503,
            "YouTube API is not configured on the server."
        )

    from urllib.parse import urlencode

    query = dict(params)
    query["key"] = YOUTUBE_API_KEY

    url = (
        "https://www.googleapis.com/youtube/v3/"
        + endpoint
        + "?"
        + urlencode(query)
    )

    request = Request(
        url,
        headers={"User-Agent": "VYRO-CLIPS/4.1"}
    )

    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    except HTTPError as error:
        try:
            details = json.loads(error.read().decode("utf-8"))
            message = details.get("error", {}).get(
                "message", "YouTube API request failed."
            )
        except Exception:
            message = "YouTube API request failed."

        raise HTTPException(error.code, message)

    except URLError:
        raise HTTPException(
            502,
            "Could not connect to YouTube. Please try again."
        )

    except Exception:
        logger.exception("YouTube API error")
        raise HTTPException(502, "YouTube API error.")


def get_youtube_metadata(video_id):
    try:
        data = youtube_api_get(
            "videos",
            {
                "part": "snippet,liveStreamingDetails,contentDetails",
                "id": video_id,
            }
        )

        items = data.get("items", [])

        if not items:
            raise HTTPException(
                404,
                "YouTube video was not found or is unavailable."
            )

        item = items[0]
        snippet = item.get("snippet", {})
        details = item.get("liveStreamingDetails", {})

        thumbnails = snippet.get("thumbnails", {})
        thumbnail = (
            thumbnails.get("high")
            or thumbnails.get("medium")
            or thumbnails.get("default")
            or {}
        )

        return {
            "video_id": video_id,
            "title": snippet.get("title", "Unknown title"),
            "channel": snippet.get("channelTitle", "Unknown channel"),
            "description": snippet.get("description", ""),
            "thumbnail": thumbnail.get("url"),
            "published_at": snippet.get("publishedAt"),
            "live_broadcast_content": snippet.get(
                "liveBroadcastContent", "none"
            ),
            "actual_start_time": details.get("actualStartTime"),
            "actual_end_time": details.get("actualEndTime"),
            "active_live_chat_id": details.get("activeLiveChatId"),
            "duration": item.get("contentDetails", {}).get("duration"),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Metadata lookup failed")
        raise HTTPException(502, "Could not inspect this YouTube video.")


# ---------------------------------------------------------
# YOUTUBE INSPECT
# ---------------------------------------------------------

class YouTubeRequest(BaseModel):
    url: str


@app.post("/youtube/inspect")
def youtube_inspect(payload: YouTubeRequest):
    video_id = extract_youtube_video_id(payload.url)
    metadata = get_youtube_metadata(video_id)

    return {
        "success": True,
        "video": metadata,
        "video_id": video_id,
        "title": metadata["title"],
        "channel": metadata["channel"],
        "thumbnail": metadata["thumbnail"],
        "url": metadata["url"],
        "footage_available": False,
        "footage_message": (
            "YouTube metadata is available. Upload matching video "
            "footage that you own or are authorised to use to create MP4 clips."
        ),
        "youtube_footage_download": False,
    }


# ---------------------------------------------------------
# YOUTUBE LIVE STATUS
# ---------------------------------------------------------

@app.post("/youtube/live-status")
def youtube_live_status(payload: YouTubeRequest):
    video_id = extract_youtube_video_id(payload.url)
    metadata = get_youtube_metadata(video_id)

    live_state = metadata.get("live_broadcast_content", "none")
    is_live = live_state == "live"
    is_upcoming = live_state == "upcoming"

    return {
        "success": True,
        "video_id": video_id,
        "title": metadata["title"],
        "channel": metadata["channel"],
        "thumbnail": metadata["thumbnail"],
        "is_live": is_live,
        "is_upcoming": is_upcoming,
        "status": (
            "LIVE" if is_live
            else "UPCOMING" if is_upcoming
            else "NOT LIVE"
        ),
        "actual_start_time": metadata.get("actual_start_time"),
        "actual_end_time": metadata.get("actual_end_time"),
        "active_live_chat_id": metadata.get("active_live_chat_id"),
        "message": (
            "This stream is currently live."
            if is_live
            else "This video is not currently live."
        ),
    }


# ---------------------------------------------------------
# YOUTUBE LIVE CHAT
# ---------------------------------------------------------

def analyse_chat_messages(messages):
    words = {}
    questions = 0
    reactions = 0
    recent_messages = []
    buckets = {}

    reaction_terms = {
        "wow", "omg", "lol", "lmao", "haha",
        "🔥", "❤️", "😂", "😭", "😱", "👏",
        "insane", "crazy", "amazing", "nice",
    }

    for item in messages:
        snippet = item.get("snippet", {})
        author = item.get("authorDetails", {})

        message = snippet.get("displayMessage", "")
        timestamp = snippet.get("publishedAt")

        if not message:
            continue

        if "?" in message:
            questions += 1

        lower = message.lower()

        if any(term in lower for term in reaction_terms):
            reactions += 1

        for word in re.findall(r"[a-zA-Z]{3,}", lower):
            if word not in {
                "the", "and", "you", "for", "this",
                "that", "are", "was", "with", "have",
                "what", "your", "just", "but", "not",
            }:
                words[word] = words.get(word, 0) + 1

        if timestamp:
            bucket = timestamp[:16]
            buckets[bucket] = buckets.get(bucket, 0) + 1

        recent_messages.append({
            "author": author.get("displayName", "Viewer"),
            "message": message,
            "published_at": timestamp,
        })

    topics = sorted(
        words.items(),
        key=lambda item: item[1],
        reverse=True
    )[:10]

    activity = [
        {"time": key, "messages": value}
        for key, value in sorted(buckets.items())
    ]

    return {
        "message_count": len(messages),
        "question_count": questions,
        "reaction_count": reactions,
        "top_topics": [
            {"topic": word, "count": count}
            for word, count in topics
        ],
        "recent_messages": recent_messages[-20:],
        "activity_timeline": activity,
    }


@app.post("/youtube/live-chat")
def youtube_live_chat(
    payload: dict
):
    url = payload.get("url", "")
    video_id = extract_youtube_video_id(url)

    metadata = get_youtube_metadata(video_id)
    chat_id = metadata.get("active_live_chat_id")

    if not chat_id:
        return {
            "success": True,
            "available": False,
            "message": (
                "Live chat is not currently available for this video. "
                "It may not be live, or chat may be disabled."
            ),
            "messages": [],
            "analysis": {
                "message_count": 0,
                "question_count": 0,
                "reaction_count": 0,
                "top_topics": [],
                "recent_messages": [],
                "activity_timeline": [],
            },
            "highlight_moments": [],
            "next_page_token": None,
        }

    params = {
        "liveChatId": chat_id,
        "part": "snippet,authorDetails",
        "maxResults": 200,
    }

    page_token = payload.get("page_token")

    if page_token:
        params["pageToken"] = page_token

    data = youtube_api_get("liveChat/messages", params)
    messages = data.get("items", [])

    analysis = analyse_chat_messages(messages)

    return {
        "success": True,
        "available": True,
        "video_id": video_id,
        "messages": messages,
        "analysis": analysis,
        "message_count": analysis["message_count"],
        "question_count": analysis["question_count"],
        "reaction_count": analysis["reaction_count"],
        "top_topics": analysis["top_topics"],
        "recent_messages": analysis["recent_messages"],
        "activity_timeline": analysis["activity_timeline"],
        "highlight_moments": [],
        "next_page_token": data.get("nextPageToken"),
        "polling_interval_millis": data.get(
            "pollingIntervalMillis", 5000
        ),
        "timestamp_basis": "chat_message_publish_time",
        "note": (
            "Chat activity is audience information, not verified "
            "video footage highlights or synchronised clip timestamps."
        ),
    }


# ---------------------------------------------------------
# VIDEO PROCESSING
# ---------------------------------------------------------

def run_command(command):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            logger.error(result.stderr[-3000:])
            raise RuntimeError("Video processing command failed.")

        return result.stdout

    except subprocess.TimeoutExpired:
        raise RuntimeError("Video processing timed out.")


def get_video_duration(path):
    output = run_command([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])

    try:
        return float(output.strip())
    except Exception:
        raise RuntimeError("Could not read video duration.")


def get_transcription(path):
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
        )

        segments, info = model.transcribe(
            str(path),
            beam_size=1,
            vad_filter=True,
        )

        results = []

        for segment in segments:
            text_value = (segment.text or "").strip()

            if text_value:
                results.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text_value,
                })

        return results

    except Exception:
        logger.exception("Transcription failed")
        return []


def score_segments(segments, duration):
    candidates = []

    if not segments:
        return candidates

    positive_terms = {
        "wow", "amazing", "incredible", "crazy",
        "unbelievable", "finally", "secret", "winner",
        "wait", "look", "listen", "important",
        "never", "best", "worst", "insane",
        "surprise", "actually", "problem",
    }

    window = 30
    step = 10

    position = 0

    while position < duration:
        start = position
        end = min(position + window, duration)

        matching = [
            segment for segment in segments
            if segment["start"] < end
            and segment["end"] > start
        ]

        if matching:
            text_value = " ".join(
                item["text"] for item in matching
            ).lower()

            word_count = len(text_value.split())

            keyword_score = sum(
                1 for term in positive_terms
                if term in text_value
            )

            question_score = text_value.count("?")
            exclamation_score = text_value.count("!")

            score = (
                min(word_count / 12, 5)
                + keyword_score * 1.5
                + question_score * 0.8
                + exclamation_score * 0.4
            )

            candidates.append({
                "start": max(0, start),
                "end": min(duration, end),
                "score": round(score, 2),
                "transcript": text_value[:500],
            })

        position += step

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selected = []

    for candidate in candidates:
        overlaps = any(
            abs(candidate["start"] - chosen["start"]) < 20
            for chosen in selected
        )

        if not overlaps:
            selected.append(candidate)

        if len(selected) >= MAX_CLIPS:
            break

    selected.sort(key=lambda item: item["start"])
    return selected


def export_clip(source, start, end, output):
    length = max(1, end - start)

    run_command([
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(source),
        "-t", str(length),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output),
    ])


def process_video(job_id, video_path):
    try:
        update_job(
            job_id,
            status="processing",
            stage="Reading video",
            progress=10,
        )

        duration = get_video_duration(video_path)

        if duration <= 0:
            raise RuntimeError("The uploaded video is empty.")

        update_job(
            job_id,
            stage="Transcribing audio",
            progress=25,
        )

        segments = get_transcription(video_path)

        update_job(
            job_id,
            stage="Finding interesting moments",
            progress=55,
        )

        candidates = score_segments(segments, duration)

        if not candidates:
            # Fallback: create a small number of useful segments
            # when speech transcription does not produce candidates.
            if duration <= 30:
                candidates = [{
                    "start": 0,
                    "end": duration,
                    "score": 1,
                    "transcript": "",
                }]
            else:
                segment_length = min(30, duration)
                count = min(
                    MAX_CLIPS,
                    max(1, int(duration // segment_length))
                )

                candidates = []

                for index in range(count):
                    start = index * segment_length

                    if start >= duration:
                        break

                    candidates.append({
                        "start": start,
                        "end": min(
                            start + segment_length,
                            duration
                        ),
                        "score": 1,
                        "transcript": "",
                    })

        exported = []

        for index, candidate in enumerate(candidates, start=1):
            filename = f"{job_id}_{index}.mp4"
            output = CLIP_DIR / filename

            export_clip(
                video_path,
                candidate["start"],
                candidate["end"],
                output,
            )

            exported.append({
                "filename": filename,
                "name": f"VYRO Clip {index}",
                "start": round(candidate["start"], 2),
                "end": round(candidate["end"], 2),
                "duration": round(
                    candidate["end"] - candidate["start"], 2
                ),
                "score": candidate["score"],
                "transcript": candidate["transcript"],
                "download_url": f"/clips/{filename}",
                "url": f"/clips/{filename}",
            })

            progress = 60 + int(
                35 * index / max(1, len(candidates))
            )

            update_job(
                job_id,
                stage=f"Exporting clip {index}",
                progress=progress,
                clips=json.dumps(exported),
            )

        update_job(
            job_id,
            status="completed",
            stage="Completed",
            progress=100,
            clips=json.dumps(exported),
        )

    except Exception as error:
        logger.exception("Job processing failed: %s", job_id)

        update_job(
            job_id,
            status="failed",
            stage="Processing failed",
            error=str(error),
        )

    finally:
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------

@app.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    source_mode: str = Form("upload"),
    youtube_url: str = Form(""),
):
    if source_mode not in {"upload", "youtube"}:
        raise HTTPException(400, "Invalid source mode.")

    if source_mode == "youtube":
        if not youtube_url.strip():
            raise HTTPException(
                400,
                "Please provide the YouTube URL."
            )

        # Validate URL. This does not download YouTube footage.
        youtube_url = canonical_youtube_url(youtube_url)

    allowed_extensions = {
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"
    }

    original_name = Path(video.filename or "video.mp4").name
    extension = Path(original_name).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            400,
            "Unsupported video format. Upload MP4, MOV, MKV, WEBM, AVI or M4V."
        )

    job_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{job_id}{extension}"

    total_bytes = 0

    try:
        with saved_path.open("wb") as destination:
            while True:
                chunk = await video.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        "Video exceeds the 250 MB upload limit."
                    )

                destination.write(chunk)

    except Exception:
        saved_path.unlink(missing_ok=True)
        raise

    finally:
        await video.close()

    if total_bytes == 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(400, "The uploaded video is empty.")

    create_job(
        job_id,
        original_name,
        source_mode,
        youtube_url or None,
    )

    executor.submit(
        process_video,
        job_id,
        saved_path,
    )

    return {
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "message": (
            "Video uploaded. VYRO is preparing your clips."
        ),
        "source_mode": source_mode,
        "youtube_url": youtube_url or None,
    }


# ---------------------------------------------------------
# JOB STATUS
# ---------------------------------------------------------

@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(404, "Job not found.")

    return {
        "success": True,
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "video_name": job["video_name"],
        "source_mode": job["source_mode"],
        "youtube_url": job["youtube_url"],
        "clips": job["clips"],
        "error": job["error"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


# ---------------------------------------------------------
# CLIP DOWNLOAD
# ---------------------------------------------------------

@app.get("/clips/{filename}")
def download_clip(filename: str):
    if (
        not filename
        or Path(filename).name != filename
        or not filename.endswith(".mp4")
    ):
        raise HTTPException(400, "Invalid clip filename.")

    clip_path = CLIP_DIR / filename

    if not clip_path.is_file():
        raise HTTPException(404, "Clip not found.")

    return FileResponse(
        path=clip_path,
        media_type="video/mp4",
        filename=filename,
    )


# ---------------------------------------------------------
# HEALTH AND FEATURES
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "VYRO CLIPS API is running",
        "version": VERSION,
        "engine": "VYRO AI Highlight Intelligence + YouTube Inspector + LivePulse",
        "features": {
            "video_upload": True,
            "direct_upload_clipping": True,
            "youtube_link_with_authorised_footage": True,
            "youtube_url_analysis": True,
            "ai_highlight_detection": True,
            "clip_export": True,
            "youtube_inspector": True,
            "youtube_live_status": True,
            "youtube_chat_reading": True,
            "youtube_footage_download": False,
            "youtube_url_only_clip_export": False,
            "chat_timestamp_video_sync": False,
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": VERSION,
        "youtube_api_configured": bool(YOUTUBE_API_KEY),
        "storage_ready": (
            UPLOAD_DIR.exists()
            and CLIP_DIR.exists()
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------
# STARTUP
# ---------------------------------------------------------

@app.on_event("startup")
def startup_event():
    init_database()

    with db_connection() as db:
        db.execute("""
            UPDATE jobs
            SET status = 'failed',
                stage = 'Server restarted',
                error = 'Processing was interrupted by a server restart.'
            WHERE status IN ('queued', 'processing')
        """)
        db.commit()

    logger.info("VYRO CLIPS %s started", VERSION)
