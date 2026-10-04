import os
import re
import json
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_SPEAKER_RE = re.compile(r"^([A-Z][A-Z0-9 _\-'\.]{1,30}):\s+(.*)$")


def _extract_speaker(dialogue: str):
    """Detect 'NAME: text' patterns common in .srt files. Returns (speaker, text)."""
    m = _SPEAKER_RE.match(dialogue.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, dialogue


def parse_srt(file_path: str, movie_title: str) -> List[Dict]:
    """
    Parses an .srt file. Returns a list of cue dicts:
        {"text": ..., "metadata": {"movie_title", "start_time", "end_time", "speaker"}}
    """
    with open(file_path, "r", encoding="utf-8-sig") as file:
        content = file.read().replace("\r\n", "\n")

    blocks = content.strip().split("\n\n")
    subtitles = []

    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 3:
            continue

        timecode_line = next((line for line in lines if "-->" in line), None)
        if not timecode_line:
            continue

        time_parts = timecode_line.split(" --> ")
        if len(time_parts) != 2:
            continue

        start_time = time_parts[0].strip()
        end_time = time_parts[1].strip()

        text_index = lines.index(timecode_line) + 1
        dialogue = " ".join(lines[text_index:])

        # Remove HTML styling like <i> or <font>
        clean_dialogue = re.sub(r"<[^>]+>", "", dialogue).strip()
        if not clean_dialogue:
            continue

        speaker, spoken_text = _extract_speaker(clean_dialogue)

        subtitles.append({
            "text": spoken_text,
            "metadata": {
                "movie_title": movie_title,
                "start_time": start_time,
                "end_time": end_time,
                "speaker": speaker or "",
            }
        })

    return subtitles


def extract_characters_with_llm(cues: List[Dict], model: str = "gpt-4o-mini") -> List[str]:
    """
    Use an LLM to extract character names from a movie's dialogue.
    Returns a sorted list of character names, or [] on failure.
    """
    if not cues:
        return []

    all_text = " ".join(c["text"] for c in cues)
    all_text = all_text[:15000]  # keep cost bounded

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = (
        "You are extracting characters from a movie's subtitles. "
        "Return ONLY a JSON object of the form: "
        '{"characters": ["Name1", "Name2", ...]}. '
        "Rules: "
        "- Include only proper names of PEOPLE who speak or are addressed in the dialogue. "
        "- Do NOT include place names, brand names, or common nouns. "
        "- Prefer first names as they appear. "
        "- Exclude the narrator and the literal word NARRATOR. "
        '- If no characters are identifiable, return {"characters": []}. '
        "- Output JSON only, no prose, no markdown fences."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": all_text},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        names = data.get("characters", [])
        return sorted({n.strip() for n in names if isinstance(n, str) and n.strip()})
    except Exception as e:
        print(f"  [warn] character extraction failed: {e}")
        return []


def chunk_subtitles(
    cues: List[Dict],
    chunk_size: int = 5,
    overlap: int = 1,
    characters: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Merge consecutive cues into overlapping chunks (~30-100 tokens).
    Each chunk records its own start/end times (first/last cue),
    unique speakers, movie characters, movie title, and a deterministic chunk_id.
    """
    if not cues:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    movie = cues[0]["metadata"]["movie_title"]
    movie_clean = movie.replace(" ", "_")
    char_str = ",".join(characters or [])

    idx = 0
    chunk_index = 0
    while idx < len(cues):
        window = cues[idx: idx + chunk_size]
        if not window:
            break

        text = " ".join(c["text"] for c in window).strip()
        if not text:
            idx += step
            continue

        speakers = sorted({c["metadata"]["speaker"] for c in window if c["metadata"]["speaker"]})

        chunks.append({
            "text": text,
            "metadata": {
                "movie_title": movie,
                "start_time": window[0]["metadata"]["start_time"],
                "end_time": window[-1]["metadata"]["end_time"],
                "speakers": ",".join(speakers),
                "characters": char_str,          # NEW
                "chunk_id": f"{movie_clean}_chunk_{chunk_index}",
                "source": "srt",
            }
        })
        chunk_index += 1
        idx += step

    return chunks


def parse_and_chunk(
    file_path: str,
    movie_title: str,
    chunk_size: int = 5,
    overlap: int = 1,
    extract_characters: bool = True,
) -> List[Dict]:
    """Parse, optionally extract characters via LLM, then chunk."""
    cues = parse_srt(file_path, movie_title)
    characters: List[str] = []
    if extract_characters and cues:
        print(f"  Extracting characters for '{movie_title}'...")
        characters = extract_characters_with_llm(cues)
        print(f"  Found {len(characters)} character(s): {characters}")
    return chunk_subtitles(cues, chunk_size=chunk_size, overlap=overlap, characters=characters)


if __name__ == "__main__":
    print("Testing parser + chunker...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_file = os.path.join(base_dir, "data", "subtitles", "The_Matrix_1999.srt")
    try:
        cues = parse_srt(sample_file, "The Matrix")
        print(f"Parsed {len(cues)} cues.")
        chars = extract_characters_with_llm(cues)
        print(f"Characters: {chars}")
        chunks = chunk_subtitles(cues, chunk_size=5, overlap=1, characters=chars)
        print(f"Produced {len(chunks)} chunks.")
        for c in chunks[:3]:
            print(c)
    except Exception as e:
        print(f"Error testing parser: {e}")