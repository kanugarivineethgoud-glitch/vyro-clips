import json
import logging
import os
import re
import sqlite3
import subprocess
import uuid
import time

from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# =========================================================
# VYRO CLIPS 4.0
# UPLOAD + YOUTUBE INSPECTOR + LIVEPULSE + CHAT
# =========================================================

VERSION = "4.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vyro")

app = FastAPI(
    title="VYRO CLIPS API",
    version=VERSION,
    description=(
        "Video highlight generation, YouTube inspection, "
        "LivePulse and live-chat analysis."
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
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}

executor = ThreadPoolExecutor(max_workers=1)
whisper_model = None

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
        raise HTTPException(400, "Please provide a YouTube URL.")

    raw_url = raw_url.strip()

    if len(raw_url) > 2048:
        raise HTTPException(400, "The supplied URL is too long.")

    try:
        parsed = urlparse(raw_url)
        if parsed.port:
            raise HTTPException(400, "Unsupported YouTube URL.")
    except ValueError:
        raise HTTPException(400, "Invalid URL.")

    if parsed.scheme.lower() != "https":
        raise HTTPException(400, "Please use an HTTPS YouTube URL.")

    if parsed.username or parsed.password:
        raise HTTPException(400, "Unsupported YouTube URL.")

    host = (parsed.hostname or "").lower()

    if host not in YOUTUBE_HOSTS:
        raise HTTPException(400, "Only supported YouTube URLs are accepted.")

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
            400,
            "Could not identify a valid YouTube video ID."
        )

    return video_id


def canonical_youtube_url(video_id):
    return (
        "https://www.youtube.com/watch?"
        + urlencode({"v": video_id})
    )


# =========================================================
# YOUTUBE API
# =========================================================

def youtube_api_get(endpoint, params):
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            503,
            "YouTube API key is not configured on the server."
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
            errors = body.get("error", {}).get("errors", [])
            if errors:
                reason = errors[0].get("reason", "")
        except Exception:
            pass

        if reason == "quotaExceeded":
            raise HTTPException(
                429,
                "YouTube API quota reached. Please try again later."
            )

        if reason in {
            "liveChatDisabled",
            "liveChatEnded",
            "liveChatNotFound"
        }:
            raise HTTPException(
                409,
                "This live chat is disabled, ended or unavailable."
            )

        if error.code == 403:
            raise HTTPException(
                403,
                "YouTube denied access. Check the API key and quota."
            )

        if error.code == 404:
            raise HTTPException(404, "YouTube resource not found.")

        raise HTTPException(502, "YouTube returned an API error.")

    except (URLError, TimeoutError):
        raise HTTPException(
            502,
            "Could not connect to YouTube. Please try again."
        )

    except (ValueError, UnicodeDecodeError):
        raise HTTPException(
            502,
            "YouTube returned an invalid response."
        )


# =========================================================
# YOUTUBE INSPECTOR
# =========================================================

def fetch_youtube_metadata(video_id):
    video_url = canonical_youtube_url(video_id)

    oembed_url = (
        "https://www.youtube.com/oembed?"
        + urlencode({"url": video_url, "format": "json"})
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
                422,
                "YouTube could not provide public information for this video."
            )
        raise HTTPException(502, "YouTube metadata service error.")

    except (URLError, TimeoutError):
        raise HTTPException(502, "Could not reach YouTube.")

    except (ValueError, UnicodeDecodeError):
        raise HTTPException(502, "YouTube returned invalid metadata.")

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
            "Video information retrieved. Upload matching footage "
            "that you own or are authorised to use to create clips."
        )
    }


# =========================================================
# LIVEPULSE
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
        raise HTTPException(404, "YouTube did not return this video.")

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
            "Live status checked. Chat availability does not provide "
            "access to the video footage."
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
# LIVE CHAT ANALYSIS
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
        ).strip()[:500]

        if not message_text:
            continue

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
                pass

        lowered = message_text.lower()

        is_question = (
            "?" in message_text
            or lowered.startswith(
                ("why ", "how ", "what ", "when ", "where ")
            )
        )

        is_reaction = bool(
            re.search(
                r"😂|🤣|😭|🔥|❤️|😍|😮|😱|👏|💀|🎉|💯",
                message_text
            )
        )

        question_count += int(is_question)
        reaction_count += int(is_reaction)

        for word in re.findall(r"[a-zA-Z]{3,}", lowered):
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

    moments.sort(key=lambda item: item["score"], reverse=True)

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
        raise HTTPException(404, "YouTube could not find this video.")

    video = video_items[0]
    snippet = video.get("snippet", {})
    details = video.get("liveStreamingDetails", {})
    chat_id = details.get("activeLiveChatId")

    if not chat_id:
        return {
            "ok": True,
            "chat_available": False,
            "live_status": snippet.get("liveBroadcastContent", "none"),
            "title": snippet.get("title"),
            "channel": snippet.get("channelTitle"),
            "message": (
                "No active live-chat ID was provided by YouTube. "
                "The stream may have ended or chat may be unavailable."
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
            raise HTTPException(400, "Invalid chat page token.")
        chat_params["pageToken"] = page_token

    chat_data = youtube_api_get(
        "liveChat/messages",
        chat_params
    )

    analysis = analyse_live_chat_messages(
        chat_data.get("items", []),
        details.get("actualStartTime")
    )

    return {
        "ok": True,
        "chat_available": True,
        "provider": "YouTube Data API v3",
        "video_id": video_id,
        "title": snippet.get("title"),
        "channel": snippet.get("channelTitle"),
        "live_status": snippet.get("liveBroadcastContent", "none"),
        "messages": analysis,
        "polling_interval_ms": chat_data.get("pollingIntervalMillis"),
        "next_page_token": chat_data.get("nextPageToken"),
        "message": (
            "Live chat page analysed. Chat timestamps are not "
            "verified video clip timestamps."
        )
    }


# =========================================================
# DATABASE
# =========================================================

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

        columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(jobs)").fetchall()
        }

        additions = {
            "stage": "TEXT DEFAULT 'queued'",
            "progress": "INTEGER DEFAULT 0",
            "timings": "TEXT DEFAULT '{}'",
            "source_mode": "TEXT DEFAULT 'upload'",
            "youtube_url": "TEXT",
            "video_name": "TEXT"
        }

        for name, definition in additions.items():
            if name not in columns:
                db.execute(
                    f"ALTER TABLE jobs ADD COLUMN {name} {definition}"
                )


def create_job(job_id, source_mode, youtube_url, video_name):
    with get_db() as db:
        db.execute(
            """
            INSERT INTO jobs (
                job_id, status, stage, progress,
                source_mode, youtube_url, video_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                "queued",
                "Waiting",
                0,
                source_mode,
                youtube_url,
                video_name
            )
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
# WHISPER
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
# HIGHLIGHT SCORING
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

FILLER_WORDS = ["um", "uh", "erm", "basically", "literally"]


def count_phrases(text, phrases):
    """Count phrase matches without matching inside unrelated words."""
    total = 0
    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        total += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return total


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
        len(re.findall(r"\b" + re.escape(filler) + r"\b", text))
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


def build_fallback_candidates(duration):
    """Create at most MAX_CLIPS segments that span the full video."""
    if duration <= 0:
        return []
    count = min(MAX_CLIPS, max(1, int(duration // 30) + (duration % 30 > 0)))
    segment_length = duration / count
    candidates = []
    for index in range(count):
        start = index * segment_length
        end = duration if index == count - 1 else (index + 1) * segment_length
        if end > start:
            candidates.append({
                "start": start,
                "end": end,
                "score": 1,
                "transcript": ""
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

        if not overlaps:
            selected.append(candidate)

        if len(selected) >= MAX_CLIPS:
            break

    return sorted(selected, key=lambda item: item["start"])


# =========================================================
# VIDEO PROCESSING
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
                logger.exception("Scene detection failed for %s", job_id)

        if not candidates:
            # No usable speech or scenes: cover the full video in up to five parts.
            candidates = build_fallback_candidates(duration)

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
                raise Exception(f"Clip {index} was not exported correctly.")

            results.append({
                "clip": index,
                "start": round(clip["start"], 2),
                "end": round(clip["end"], 2),
                "score": clip["score"],
                "transcript": clip["transcript"],
                "download": f"/clips/{output.name}"
            })

            progress = 65 + int(index / len(selected) * 30)

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

        update_job(
            job_id,
            status="completed",
            stage="Completed",
            progress=100,
            clips=results,
            timings=timings
        )

        logger.info("Job %s completed: %s", job_id, timings)

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
            "direct_upload_clipping": True,
            "youtube_link_with_authorised_footage": True,
            "ai_highlight_detection": True,
            "clip_export": True,
            "youtube_inspector": True,
            "youtube_live_status": True,
            "youtube_chat_reading": True,
            "youtube_footage_download": False,
            "chat_timestamp_video_sync": False
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
async def upload_video(
    video: UploadFile = File(...),
    source_mode: str = Form("upload"),
    youtube_url: str = Form("")
):
    filename = video.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            "Unsupported video format. Use MP4, MOV, MKV or WEBM."
        )

    if source_mode not in {"upload", "youtube"}:
        raise HTTPException(400, "Invalid video source mode.")

    youtube_url = youtube_url.strip()

    if source_mode == "youtube":
        if not youtube_url:
            raise HTTPException(
                400,
                "Please provide the YouTube link for this footage."
            )
        extract_youtube_video_id(youtube_url)

    elif youtube_url:
        extract_youtube_video_id(youtube_url)

    job_id = str(uuid.uuid4())
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
                        413,
                        "Video exceeds the 250 MB upload limit."
                    )

                buffer.write(chunk)

        if total_size == 0:
            raise HTTPException(400, "Uploaded video is empty.")

        create_job(
            job_id,
            source_mode,
            youtube_url or None,
            filename[:255]
        )

        executor.submit(
            analyse_video,
            job_id,
            video_path
        )

        return {
            "job_id": job_id,
            "status": "queued",
            "source_mode": source_mode,
            "youtube_url": youtube_url or None,
            "video_name": filename[:255],
            "message": "Video uploaded. VYRO analysis started."
        }

    except HTTPException:
        video_path.unlink(missing_ok=True)
        raise

    except Exception as error:
        video_path.unlink(missing_ok=True)
        logger.exception("Upload failed")
        raise HTTPException(500, f"Upload failed: {str(error)}")

    finally:
        await video.close()


@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)

    if job is None:
        raise HTTPException(404, "Job not found.")

    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "clips": json.loads(job["clips"]),
        "stage": job["stage"],
        "progress": job["progress"],
        "timings": json.loads(job["timings"] or "{}"),
        "source_mode": job["source_mode"],
        "youtube_url": job["youtube_url"],
        "video_name": job["video_name"]
    }

    if job["error"]:
        response["error"] = job["error"]

    return response


@app.get("/clips/{filename}")
def download_clip(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(400, "Invalid filename.")

    if not filename.lower().endswith(".mp4"):
        raise HTTPException(400, "Invalid clip file.")

    path = OUTPUTS / filename

    if not path.is_file():
        raise HTTPException(404, "Clip not found.")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename
    )
