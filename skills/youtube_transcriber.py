"""YouTube transcript extraction.

Uses :mod:`youtube_transcript_api` for captions and :mod:`pytube` for
metadata. Both libraries break occasionally because YouTube changes its
internals — every method here returns sensible defaults rather than
raising.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger


_VIDEO_ID_RE = re.compile(
    r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_\-]{11})"
)


@dataclass
class YoutubeVideo:
    id: str
    title: str
    transcript: str
    url: str
    duration: int  # seconds
    channel: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "transcript": self.transcript,
            "url": self.url,
            "duration": self.duration,
            "channel": self.channel,
        }


def _extract_video_id(url_or_id: str) -> str:
    if not url_or_id:
        return ""
    if len(url_or_id) == 11 and re.fullmatch(r"[A-Za-z0-9_\-]{11}", url_or_id):
        return url_or_id
    match = _VIDEO_ID_RE.search(url_or_id)
    return match.group(1) if match else ""


class YoutubeTranscriber:
    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages or ["en", "fr", "es", "de"]

    def get_transcript(self, url: str) -> dict[str, Any]:
        video_id = _extract_video_id(url)
        if not video_id:
            return {"id": "", "title": "", "transcript": "", "url": url, "duration": 0, "channel": ""}

        transcript_text = self._fetch_transcript(video_id)
        title, channel, duration = self._fetch_metadata(video_id)

        result = YoutubeVideo(
            id=video_id,
            title=title,
            transcript=transcript_text,
            url=f"https://www.youtube.com/watch?v={video_id}",
            duration=duration,
            channel=channel,
        )
        return result.as_dict()

    def _fetch_transcript(self, video_id: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:  # pragma: no cover
            logger.error("youtube-transcript-api is not installed.")
            return ""
        try:
            entries = YouTubeTranscriptApi.get_transcript(video_id, languages=self.languages)
            return " ".join(e.get("text", "").strip() for e in entries if e.get("text")).strip()
        except Exception as exc:
            logger.debug("Transcript unavailable for {}: {}", video_id, exc)
            return ""

    def _fetch_metadata(self, video_id: str) -> tuple[str, str, int]:
        try:
            from pytube import YouTube
        except ImportError:  # pragma: no cover
            return "", "", 0
        try:
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            return (yt.title or ""), (yt.author or ""), int(yt.length or 0)
        except Exception as exc:
            logger.debug("Metadata unavailable for {}: {}", video_id, exc)
            return "", "", 0

    # --------------------------------------------------------------- search

    def search_youtube(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Light-weight YouTube search via DuckDuckGo (avoids YouTube API key)."""
        if not query.strip():
            return []
        try:
            from duckduckgo_search import DDGS
        except ImportError:  # pragma: no cover
            return []
        results: list[dict[str, Any]] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.videos(query, max_results=max_results):
                    url = item.get("content") or item.get("url") or ""
                    if "youtube.com" not in url and "youtu.be" not in url:
                        continue
                    vid = _extract_video_id(url)
                    if not vid:
                        continue
                    results.append(
                        {
                            "id": vid,
                            "title": item.get("title", ""),
                            "url": url,
                            "channel": item.get("uploader", ""),
                            "duration": item.get("duration", ""),
                            "description": item.get("description", ""),
                        }
                    )
        except Exception as exc:
            logger.warning("YouTube search failed: {}", exc)
        return results[:max_results]
