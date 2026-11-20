import json
import logging
import os
import re
import sqlite3
import subprocess
import time
import uuid

from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# =========================================================
# VYRO CLIPS 3.4
# AI HIGHLIGHTS + YOUTUBE INSPECTOR + LIVEPULSE + CHAT
# =========================================================

VERSION = "3.4"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vyro")

app = FastAPI(
    title="VYRO CLIPS API",
    version=VERSION,
    description=(
        "AI video highlight generation, YouTube metadata, "
        "live status and accessible live-chat analysis."
    )
)

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
MAX_CLIPS = 5

executor = ThreadPoolExecutor(max_workers=1)
whisper_model = None

# Keep this key only in Render environment variables.
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


# =========================================================
# REQUEST MODELS
# =========================================================

class YouTubeRequest(BaseModel):
    url: str


class YouTubeLiveChatRequest(BaseModel):
    url: str
    page_token: str | None = None


# =========================================================
# YOUTUBE URL VALIDATION
# =========================================================

def extract_youtube_video_id(raw_url):
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide a YouTube video URL."
        )

    raw_url = raw_url.strip()

    if len(raw_url) > 2048:
        raise HTTPException(
            status_code=400,
            detail="The supplied URL is too long."
        )

    try:
        parsed = urlparse(raw_url)

        if parsed.port:
            raise HTTPException(
                status_code=400,
                detail="This YouTube URL format is not supported."
            )

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid URL.")

    if parsed.scheme.lower() != "https":
        raise HTTPException(
            status_code=400,
            detail="Please use a secure HTTPS YouTube URL."
        )

    if parsed.username or parsed.password:
        raise HTTPException(
            status_code=400,
            detail="This YouTube URL format is not supported."
        )

    host = (parsed.hostname or "").lower()

    if host not in YOUTUBE_HOSTS:
        raise HTTPException(
            status_code=400,
            detail="Only supported YouTube URLs are accepted."
        )

    parts = [part for part in parsed.path.split("/") if part]
    video_id = None

    if host == "youtu.be":
        if parts:
            video_id = parts[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    elif len(parts) >= 2 and parts[0] in {
        "shorts", "live", "embed", "v"
    }:
        video_id = parts[1]

    if not video_id or not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not identify a valid YouTube video ID. "
                "Use a supported video, Shorts or live-video URL."
            )
        )

    return video_id


def canonical_youtube_url(video_id):
    return (
        "https://www.youtube.com/watch?"
        + urlencode({"v": video_id})
    )


# =========================================================
# YOUTUBE HTTP REQUESTS
# =========================================================

def youtube_api_get(endpoint, params):
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "YouTube API key is not configured on the server. "
                "Check YOUTUBE_API_KEY in Render environment variables."
            )
        )

    request_params = dict(params)
    request_params["key"] = YOUTUBE_API_KEY

    api_url = (
        "https://www.googleapis.com/youtube/v3/"
        + endpoint
        + "?"
        + urlencode(request_params)
    )

    request = Request(
        api_url,
        headers={"User-Agent": f"VYRO-CLIPS/{VERSION}"}
    )

    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(
                response.read(2 * 1024 * 1024).decode("utf-8")
            )

    except HTTPError as error:
        reason = ""

        try:
            body = json.loads(
                error.read(256 * 1024).decode("utf-8")
            )
            api_error = body.get("error", {})
            errors = api_error.get("errors", [])
            if errors:
                reason = errors[0].get("reason", "")
        except Exception:
            pass

        logger.warning(
            "YouTube API error: HTTP %s, reason=%s",
            error.code,
            reason
        )

        if reason == "quotaExceeded":
            raise HTTPException(
                status_code=429,
                detail=(
                    "YouTube API quota has been reached. "
                    "Please try again after the quota resets."
                )
            )

        if reason in {
            "liveChatDisabled",
            "liveChatEnded",
            "liveChatNotFound"
        }:
            raise HTTPException(
                status_code=409,
                detail=(
                    "This live chat is disabled, has ended "
                    "or is unavailable."
                )
            )

        if error.code == 403:
            raise HTTPException(
                status_code=403,
                detail=(
                    "YouTube denied access. Check API key restrictions, "
                    "API activation and available quota."
                )
            )

        if error.code == 404:
            raise HTTPException(
                status_code=404,
                detail="The requested YouTube resource was not found."
            )

        raise HTTPException(
            status_code=502,
            detail="YouTube returned an API error. Please try again."
        )

    except (URLError, TimeoutError):
        raise HTTPException(
            status_code=502,
            detail="Could not connect to YouTube. Please try again."
        )

    except (ValueError, UnicodeDecodeError):
        raise HTTPException(
            status_code=502,
            detail="YouTube returned an invalid response."
        )


# =========================================================
# YOUTUBE INSPECTOR
# =========================================================

def fetch_youtube_metadata(video_id):
    video_url = canonical_youtube_url(video_id)

    oembed_url = (
        "https://www.youtube.com/oembed?"
        + urlencode({
            "url": video_url,
            "format": "json"
        })
    )

    request = Request(
        oembed_url,
        headers={"User-Agent": f"VYRO-CLIPS/{VERSION}"}
    )

    try:
        with urlopen(request, timeout=12) as response:
            payload = json.loads(
                response.read(1024 * 1024).decode("utf-8")
            )

    except HTTPError as error:
        if error.code in (400, 401, 403, 404):
            raise HTTPException(
                status_code=422,
                detail=(
                    "YouTube could not provide public information "
                    "for this video. It may be private, restricted "
                    "or unavailable."
                )
            )

        raise HTTPException(
            status_code=502,
            detail="YouTube metadata service returned an error."
        )

    except (URLError, TimeoutError):
        raise HTTPException(
            status_code=502,
            detail="Could not reach YouTube. Please try again."
        )

    except (ValueError, UnicodeDecodeError):
        raise HTTPException(
            status_code=502,
            detail="YouTube returned invalid metadata."
        )

    return {
        "ok": True,
        "provider": "YouTube",
        "video_id": video_id,
        "url": video_url,
        "title": payload.get("title"),
        "channel": payload.get("author_name"),
        "channel_url": payload.get("author_url"),
        "thumbnail": payload.get("thumbnail_url"),
        "live_status": "unknown",
        "live_chat": "not_checked",
        "footage_access": "not_provided",
        "message": (
            "Public video information retrieved. "
            "Live status and chat have not been checked. "
            "This endpoint does not retrieve video footage."
        )
    }


# =========================================================
# LIVEPULSE — LIVE STATUS
# =========================================================

def fetch_youtube_live_status(video_id):
    data = youtube_api_get(
        "videos",
        {
            "part": "snippet,liveStreamingDetails",
            "id": video_id
        }
    )

    items = data.get("items", [])

    if not items:
        raise HTTPException(
            status_code=404,
            detail="YouTube did not return this video."
        )

    item = items[0]
    snippet = item.get("snippet", {})
    details = item.get("liveStreamingDetails", {})

    broadcast = snippet.get("liveBroadcastContent", "none")

    if broadcast == "live":
        status = "live"
    elif broadcast == "upcoming":
        status = "upcoming"
    else:
        status = "not_live"

    active_chat_id = details.get("activeLiveChatId")

    result = {
        "ok": True,
        "provider": "YouTube Data API v3",
        "video_id": video_id,
        "url": canonical_youtube_url(video_id),
        "title": snippet.get("title"),
        "channel": snippet.get("channelTitle"),
        "live_status": status,
        "live_chat_available": bool(active_chat_id),
        "live_chat_messages": "not_read",
        "footage_access": "not_provided",
        "message": (
            "Live status checked using YouTube Data API. "
            "This endpoint does not read chat messages "
            "or retrieve video footage."
        )
    }

    for source_key, output_key in (
        ("actualStartTime", "actual_start_time"),
        ("scheduledStartTime", "scheduled_start_time"),
        ("concurrentViewers", "concurrent_viewers"),
    ):
        if details.get(source_key) is not None:
            result[output_key] = details[source_key]

    return result


# =========================================================
# LIVE CHAT INTELLIGENCE
# =========================================================

def analyse_live_chat_messages(items, actual_start_time=None):
    stop_words = {
        "the", "and", "for", "you", "are", "this", "that",
        "with", "have", "what", "when", "where", "why",
        "how", "who", "was", "were", "will", "would",
        "can", "could", "should", "just", "not", "but",
        "all", "they", "them", "his", "her", "she", "him",
        "its", "our", "your", "from", "there", "here",
        "about", "then", "than", "too", "very", "lol",
        "bro", "guys", "yeah", "yes", "no", "ok", "okay",
        "omg", "www", "http", "https"
    }

    messages = []
    topics = Counter()
    buckets = {}

    question_count = 0
    reaction_count = 0
    start_dt = None

    if actual_start_time:
        try:
            start_dt = datetime.fromisoformat(
                actual_start_time.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            start_dt = None

    for item in items:
        snippet = item.get("snippet", {})

        message_text = (
            snippet.get("displayMessage")
            or snippet.get("textMessageDetails", {}).get(
                "messageText"
            )
            or ""
        )

        message_text = re.sub(
            r"\s+", " ", str(message_text)
        ).strip()

        if not message_text:
            continue

        message_text = message_text[:500]

        published_at = snippet.get("publishedAt")
        relative_seconds = None

        if published_at and start_dt:
            try:
                published_dt = datetime.fromisoformat(
                    published_at.replace("Z", "+00:00")
                )
                relative_seconds = max(
                    0,
                    int((published_dt - start_dt).total_seconds())
                )
            except (ValueError, TypeError):
                relative_seconds = None

        lowered = message_text.lower()

        is_question = (
            "?" in message_text
            or lowered.startswith(
                ("why ", "how ", "what ", "when ", "where ")
            )
        )

        is_reaction = bool(
            re.search(
                r"!|😂|🤣|😭|🔥|❤️|😍|😮|😱|👏|💀|🎉|💯",
                message_text
            )
        )

        question_count += int(is_question)
        reaction_count += int(is_reaction)

        words = re.findall(
            r"[a-zA-Z]{3,}",
            lowered
        )

        for word in words:
            if word not in stop_words:
                topics[word] += 1

        if relative_seconds is not None:
            bucket_start = (relative_seconds // 30) * 30

            bucket = buckets.setdefault(
                bucket_start,
                {
                    "start_seconds": bucket_start,
                    "end_seconds": bucket_start + 30,
                    "messages": 0,
                    "questions": 0,
                    "reactions": 0
                }
            )

            bucket["messages"] += 1
            bucket["questions"] += int(is_question)
            bucket["reactions"] += int(is_reaction)

        messages.append({
            "text": message_text,
            "published_at": published_at,
            "is_question": is_question,
            "is_reaction": is_reaction,
            "relative_seconds": relative_seconds
        })

    moments = []

    for bucket in buckets.values():
        score = (
            bucket["messages"]
            + bucket["questions"] * 1.5
            + bucket["reactions"] * 0.75
        )

        moments.append({
            **bucket,
            "score": round(score, 2)
        })

    moments.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    top_topics = [
        {"word": word, "count": count}
        for word, count in topics.most_common(10)
        if count >= 2
    ]

    return {
        "message_count": len(messages),
        "question_count": question_count,
        "reaction_count": reaction_count,
        "top_topics": top_topics,
        "highlight_moments": moments[:8],
        "sample_messages": messages[-30:]
    }


def fetch_youtube_live_chat(video_id, page_token=None):
    video_data = youtube_api_get(
        "videos",
        {
            "part": "snippet,liveStreamingDetails",
            "id": video_id
        }
    )

    video_items = video_data.get("items", [])

    if not video_items:
        raise HTTPException(
            status_code=404,
            detail="YouTube could not find this video."
        )

    video = video_items[0]
    snippet = video.get("snippet", {})
    details = video.get("liveStreamingDetails", {})

    chat_id = details.get("activeLiveChatId")

    if not chat_id:
        return {
            "ok": True,
            "chat_available": False,
            "live_status": snippet.get(
                "liveBroadcastContent", "none"
            ),
            "title": snippet.get("title"),
            "channel": snippet.get("channelTitle"),
            "message": (
                "YouTube did not provide an active live-chat ID. "
                "The stream may not be live or chat may be unavailable."
            ),
            "messages": {
                "message_count": 0,
                "question_count": 0,
                "reaction_count": 0,
                "top_topics": [],
                "highlight_moments": [],
                "sample_messages": []
            }
        }

    chat_params = {
        "liveChatId": chat_id,
        "part": "snippet",
        "maxResults": 200
    }

    if page_token:
        if len(page_token) > 2048:
            raise HTTPException(
                status_code=400,
                detail="Invalid chat page token."
            )

        chat_params["pageToken"] = page_token

    chat_data = youtube_api_get(
        "liveChat/messages",
        chat_params
    )

    items = chat_data.get("items", [])

    analysis = analyse_live_chat_messages(
        items,
        details.get("actualStartTime")
    )

    return {
        "ok": True,
        "chat_available": True,
        "provider": "YouTube Data API v3",
        "video_id": video_id,
        "title": snippet.get("title"),
        "channel": snippet.get("channelTitle"),
        "live_status": snippet.get(
            "liveBroadcastContent", "none"
        ),
        "messages": analysis,
        "polling_interval_ms": chat_data.get(
            "pollingIntervalMillis"
        ),
        "next_page_token": chat_data.get("nextPageToken"),
        "message": (
            "Live chat page analysed. VYRO does not store "
            "chat messages or author identities."
        )
    }


# =========================================================
# DATABASE
# =========================================================

def get_db():
    connection = sqlite3.connect(
        DATABASE,
        timeout=30
    )
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

        columns = {
            row["name"]
            for row in db.execute(
                "PRAGMA table_info(jobs)"
            ).fetchall()
        }

        additions = {
            "stage": "TEXT DEFAULT 'queued'",
            "progress": "INTEGER DEFAULT 0",
            "timings": "TEXT DEFAULT '{}'"
        }

        for name, definition in additions.items():
            if name not in columns:
                db.execute(
                    f"ALTER TABLE jobs ADD COLUMN {name} {definition}"
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
    updates = {
        "status": status,
        "clips": json.dumps(clips) if clips is not None else None,
        "error": str(error)[:2000] if error is not None else None,
        "stage": stage,
        "progress": progress,
        "timings": json.dumps(timings) if timings is not None else None
    }

    fields = []
    values = []

    for field, value in updates.items():
        if value is not None:
            fields.append(f"{field} = ?")
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


# =========================================================
# WHISPER MODEL
# =========================================================

def get_whisper():
    global whisper_model

    if whisper_model is None:
        logger.info("Loading VYRO Whisper model")

        whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            num_workers=1
        )

        logger.info("Whisper model ready")

    return whisper_model


# =========================================================
# VIDEO UTILITIES
# =========================================================

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

    duration = float(result.stdout.strip())

    if duration <= 0:
        raise Exception("Invalid video duration.")

    return duration


def detect_scenes(video_path):
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=27.0))
    manager.detect_scenes(video)
    return manager.get_scene_list()


# =========================================================
# AI HIGHLIGHT SCORING
# =========================================================

STRONG_PHRASES = [
    "no way", "oh my god", "i can't believe",
    "unbelievable", "let's go", "that's insane",
    "what just happened", "wait for it", "finally",
    "the best part", "the worst part", "you won't believe",
    "this is crazy", "that was amazing", "incredible",
    "what a moment"
]

STORY_PHRASES = [
    "but then", "because of that", "what happened next",
    "in the end", "as a result", "the truth is",
    "i realized", "i learned", "my biggest mistake",
    "the solution", "the problem was", "the reason is",
    "here is what happened", "the moment when"
]

EDUCATIONAL_PHRASES = [
    "here is how", "how to", "step by step",
    "one important thing", "a simple trick", "the secret",
    "you should know", "the difference is",
    "the important thing", "for example",
    "the main reason", "remember this"
]

EMOTION_WORDS = [
    "amazing", "awesome", "terrible", "shocking",
    "surprising", "exciting", "funny", "hilarious",
    "incredible", "beautiful", "fantastic", "ridiculous",
    "unbelievable", "serious", "dangerous"
]

FILLER_WORDS = [
    "um", "uh", "erm", "basically", "literally"
]


def count_phrases(text, phrases):
    return sum(text.count(phrase) for phrase in phrases)


def score_highlight(transcript, relevant, window_duration):
    text = transcript.lower().strip()
    words = re.findall(r"\b[\w']+\b", text)

    if not words:
        return 0

    score = 1.0

    score += min(count_phrases(text, STRONG_PHRASES), 3) * 2.5
    score += min(count_phrases(text, STORY_PHRASES), 4) * 2.0
    score += min(count_phrases(text, EDUCATIONAL_PHRASES), 4) * 1.8
    score += min(count_phrases(text, EMOTION_WORDS), 4) * 1.2

    score += min(transcript.count("?"), 3) * 0.7
    score += min(transcript.count("!"), 3) * 0.7

    spoken_seconds = sum(
        max(0, item["end"] - item["start"])
        for item in relevant
    )

    speech_ratio = min(
        1.0,
        spoken_seconds / max(window_duration, 1)
    )

    score += speech_ratio * 2

    word_count = len(words)

    if 25 <= word_count <= 110:
        score += 2
    elif 15 <= word_count < 25:
        score += 1
    elif word_count < 8:
        score -= 2

    unique_ratio = len(set(words)) / max(word_count, 1)

    if unique_ratio > 0.55:
        score += 1
    elif unique_ratio < 0.25:
        score -= 1.5

    filler_count = sum(
        len(re.findall(
            r"\b" + re.escape(filler) + r"\b",
            text
        ))
        for filler in FILLER_WORDS
    )

    if filler_count / max(word_count, 1) > 0.15:
        score -= 1.5

    if window_duration < 8:
        score -= 2

    return round(max(0, score), 2)


def build_candidates(speech, duration):
    candidates = []

    if not speech or duration <= 0:
        return candidates

    target_length = min(35, max(12, duration * 0.32))
    target_length = min(target_length, duration)

    step = max(5, target_length * 0.5)

    starts = []
    current = 0.0

    while current + target_length <= duration:
        starts.append(current)
        current += step

    final_start = max(0, duration - target_length)

    if not starts or abs(starts[-1] - final_start) > 1:
        starts.append(final_start)

    for window_start in starts:
        window_end = min(duration, window_start + target_length)

        relevant = [
            item for item in speech
            if item["end"] > window_start
            and item["start"] < window_end
        ]

        if not relevant:
            continue

        transcript = " ".join(
            item["text"] for item in relevant
        ).strip()

        score = score_highlight(
            transcript,
            relevant,
            window_end - window_start
        )

        if score <= 0:
            continue

        first_speech = relevant[0]["start"]
        last_speech = relevant[-1]["end"]

        clip_start = max(window_start, first_speech - 1.5)
        clip_end = min(window_end, last_speech + 1.5)

        if clip_end - clip_start < 10:
            clip_start = window_start
            clip_end = window_end

        if clip_end > clip_start:
            candidates.append({
                "start": round(clip_start, 3),
                "end": round(clip_end, 3),
                "score": score,
                "transcript": transcript
            })

    return candidates


def select_best_candidates(candidates):
    ranked = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True
    )

    selected = []

    for candidate in ranked:
        overlaps = False

        for existing in selected:
            overlap_start = max(
                candidate["start"],
                existing["start"]
            )
            overlap_end = min(
                candidate["end"],
                existing["end"]
            )

            overlap = max(0, overlap_end - overlap_start)

            shorter = min(
                candidate["end"] - candidate["start"],
                existing["end"] - existing["start"]
            )

            overlap_ratio = overlap / shorter if shorter > 0 else 0

            if overlap_ratio > 0.35:
                overlaps = True
                break

            if (
                candidate["start"] < existing["end"] + 2
                and candidate["end"] > existing["start"] - 2
            ):
                overlaps = True
                break

        if not overlaps:
            selected.append(candidate)

        if len(selected) >= MAX_CLIPS:
            break

    return sorted(selected, key=lambda item: item["start"])


# =========================================================
# VIDEO ANALYSIS AND CLIP EXPORT
# =========================================================

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
            progress=5,
            timings=timings
        )

        duration = get_video_duration(video_path)

        model_start = time.perf_counter()
        model = get_whisper()

        timings["model_load_seconds"] = round(
            time.perf_counter() - model_start,
            2
        )

        update_job(
            job_id,
            stage="Understanding speech",
            progress=15,
            timings=timings
        )

        transcription_start = time.perf_counter()

        segments, info = model.transcribe(
            str(video_path),
            beam_size=1,
            best_of=1,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=True
        )

        speech = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                speech.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text
                })

        timings["transcription_seconds"] = round(
            time.perf_counter() - transcription_start,
            2
        )

        logger.info(
            "Job %s transcription completed in %.2f seconds",
            job_id,
            timings["transcription_seconds"]
        )

        update_job(
            job_id,
            stage="Finding interesting moments",
            progress=55,
            timings=timings
        )

        selection_start = time.perf_counter()

        candidates = build_candidates(speech, duration)

        if not candidates and duration >= 120:
            try:
                update_job(
                    job_id,
                    stage="Checking scene changes",
                    progress=58
                )

                scenes = detect_scenes(video_path)

                for scene_start, scene_end in scenes:
                    start = max(0, scene_start.get_seconds())
                    end = min(duration, scene_end.get_seconds())

                    if end - start >= 10:
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

        if not candidates:
            candidates.append({
                "start": 0,
                "end": min(duration, 30),
                "score": 1,
                "transcript": ""
            })

        selected = select_best_candidates(candidates)

        timings["selection_seconds"] = round(
            time.perf_counter() - selection_start,
            2
        )

        if not selected:
            raise Exception("No clips could be selected.")

        update_job(
            job_id,
            stage="Creating video clips",
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
                    f"Clip {index} was not exported correctly."
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
                index / len(selected) * 30
            )

            update_job(
                job_id,
                stage=f"Exporting clip {index} of {len(selected)}",
                progress=min(progress, 95)
            )

        timings["export_seconds"] = round(
            time.perf_counter() - export_start,
            2
        )

        if not results:
            raise Exception("No clips could be generated.")

        timings["total_seconds"] = round(
            time.perf_counter() - started,
            2
        )

        logger.info(
            "Job %s completed: %s",
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
        logger.exception("Processing failed for job %s", job_id)

        update_job(
            job_id,
            status="failed",
            stage="Failed",
            error=str(error)
        )


# =========================================================
# API ROUTES
# =========================================================

@app.get("/")
def home():
    return {
        "message": "VYRO CLIPS API is running",
        "version": VERSION,
        "engine": (
            "VYRO AI Highlight Intelligence + YouTube Inspector "
            "+ LivePulse + Live Chat Intelligence"
        ),
        "features": {
            "video_upload": True,
            "ai_highlight_detection": True,
            "clip_export": True,
            "youtube_inspector": True,
            "youtube_live_status": True,
            "youtube_chat_reading": True,
            "youtube_footage_download": False
        }
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "VYRO CLIPS API",
        "version": VERSION
    }


@app.post("/youtube/inspect")
def inspect_youtube_video(request: YouTubeRequest):
    video_id = extract_youtube_video_id(request.url)
    return fetch_youtube_metadata(video_id)


@app.post("/youtube/live-status")
def youtube_live_status(request: YouTubeRequest):
    video_id = extract_youtube_video_id(request.url)
    return fetch_youtube_live_status(video_id)


@app.post("/youtube/live-chat")
def youtube_live_chat(request: YouTubeLiveChatRequest):
    video_id = extract_youtube_video_id(request.url)

    return fetch_youtube_live_chat(
        video_id,
        request.page_token
    )


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    filename = video.filename or ""

    if not filename.lower().endswith(
        (".mp4", ".mov", ".mkv", ".webm")
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
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
                        detail="Video exceeds the 250 MB upload limit."
                    )

                buffer.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty."
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
            "message": (
                "Video uploaded. VYRO AI analysis started."
            )
        }

    except HTTPException:
        video_path.unlink(missing_ok=True)
        raise

    except Exception as error:
        video_path.unlink(missing_ok=True)

        logger.exception("Upload failed")

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
            detail="Job not found."
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
            detail="Invalid filename."
        )

    path = OUTPUTS / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Clip not found."
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename
)
