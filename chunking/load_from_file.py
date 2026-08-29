import json
from pathlib import Path
from typing import Any, Tuple


def load_agenda_chunks(file_path: str | Path) -> list[dict[str, Any]]:
    with Path(file_path).open(encoding="utf-8") as file:
        data = json.load(file)

    event = data["event"]
    chunks = []

    chunks.append({
        "id": "event_overview",
        "text": (
            f"Event: {event['name']}\n"
            f"Venue: {event['venue']}\n"
            f"Dates: {event['dates']}"
        ),
        "metadata": {
            "document_type": "event",
            "event_name": event["name"],
        },
    })

    for session in data.get("sessions", []):
        speakers = ", ".join(session.get("speakers", []))

        text = (
            f"Session: {session['title']}\n"
            f"Event: {event['name']}\n"
            f"Track: {session['track']}\n"
            f"Date: {session['day']}\n"
            f"Time: {session['start']}–{session['end']}\n"
            f"Room: {session['room']}\n"
            f"Speakers: {speakers or 'Not specified'}\n"
            f"Description: {session['abstract']}"
        )

        chunks.append({
            "id": session["id"],
            "text": text,
            "metadata": {
                "document_type": "session",
                "event_name": event["name"],
                "session_id": session["id"],
                "title": session["title"],
                "track": session["track"],
                "day": session["day"],
                "start": session["start"],
                "end": session["end"],
                "room": session["room"],
                "speakers": session.get("speakers", []),
            },
        })

    for exhibitor in data.get("exhibitors", []):
        text = (
            f"Exhibitor: {exhibitor['name']}\n"
            f"Event: {event['name']}\n"
            f"Category: {exhibitor['category']}\n"
            f"Stand: {exhibitor['stand']}\n"
            f"Description: {exhibitor['description']}"
        )

        chunks.append({
            "id": exhibitor["id"],
            "text": text,
            "metadata": {
                "document_type": "exhibitor",
                "event_name": event["name"],
                "exhibitor_id": exhibitor["id"],
                "name": exhibitor["name"],
                "category": exhibitor["category"],
                "stand": exhibitor["stand"],
            },
        })

    return chunks