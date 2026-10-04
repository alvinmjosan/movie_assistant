import os
import tempfile

from ingestion.srt_parser import parse_srt, chunk_subtitles, parse_and_chunk

SAMPLE = """1
00:00:01,000 --> 00:00:03,000
NEO: Wake up, Neo...

2
00:00:04,000 --> 00:00:06,500
TRINITY: The Matrix has you.

3
00:00:07,000 --> 00:00:09,000
<i>Follow the white rabbit.</i>

4
00:00:10,000 --> 00:00:12,000
NEO: Knock, knock.

5
00:00:13,000 --> 00:00:15,000
MORPHEUS: Unfortunately, no one can be told what the Matrix is.
"""


def _write_tmp():
    fd, path = tempfile.mkstemp(suffix=".srt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(SAMPLE)
    return path


def test_parse_srt_extracts_cues_and_speakers():
    path = _write_tmp()
    try:
        cues = parse_srt(path, "The Matrix")
        assert len(cues) == 5
        assert cues[0]["metadata"]["speaker"] == "NEO"
        assert cues[1]["metadata"]["speaker"] == "TRINITY"
        assert cues[2]["text"] == "Follow the white rabbit."  # HTML stripped
        assert cues[0]["metadata"]["movie_title"] == "The Matrix"
        assert cues[0]["metadata"]["start_time"] == "00:00:01,000"
    finally:
        os.remove(path)


def test_chunking_with_overlap():
    path = _write_tmp()
    try:
        chunks = parse_and_chunk(path, "The Matrix", chunk_size=3, overlap=1)
        # 5 cues, step=2 -> start indices 0,2,4 -> 3 chunks (last has 1 cue)
        assert len(chunks) == 3
        assert chunks[0]["metadata"]["start_time"] == "00:00:01,000"
        assert chunks[1]["metadata"]["start_time"] == "00:00:07,000"
        assert "chunk_id" in chunks[0]["metadata"]
    finally:
        os.remove(path)