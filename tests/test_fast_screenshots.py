"""Faster screenshots: shrink, tile tall images, read in parallel, fewer Nemotron calls."""
import threading
import time
from io import BytesIO

import pytest
from PIL import Image

from parlaylab import slip_reader
from parlaylab.slip_reader import prepare_image, read_screenshots


def _png(w, h):
    buf = BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_normal_phone_screenshot_is_shrunk_to_one_jpeg():
    tiles = prepare_image(_png(1290, 2796))          # iPhone Pro size
    assert len(tiles) == 1 and tiles[0][1] == "image/jpeg"
    img = Image.open(BytesIO(tiles[0][0]))
    assert img.width == 1080


def test_tall_scrolling_screenshot_is_cut_into_overlapping_tiles():
    tiles = prepare_image(_png(1080, 9000))
    sizes = [Image.open(BytesIO(b)).size for b, _ in tiles]
    tile_h = round(1080 * slip_reader.TILE_RATIO)
    assert len(tiles) >= 4 and all(s == (1080, tile_h) for s in sizes)
    # tiles overlap and cover the whole image
    step = round(tile_h * (1 - slip_reader.TILE_OVERLAP))
    assert step < tile_h and (len(tiles) - 1) * step + tile_h >= 9000 - step


def test_unreadable_file_is_passed_through():
    assert prepare_image(b"not an image", "image/webp") == [(b"not an image", "image/webp")]


@pytest.fixture
def fake_parsers(monkeypatch):
    calls = {"one": [], "many": []}
    monkeypatch.setattr(slip_reader, "pick_model", lambda role, api_key=None: f"{role}-model")

    def one(text, api_key=None, book=None):
        calls["one"].append(text)
        return {"book": book, "legs": [{"team": text.split()[0], "type": "moneyline"}], "_models": {"parser": "p"}}

    def many(text, api_key=None, book=None):
        calls["many"].append(text)
        return [{"book": book, "legs": [{"team": "Knicks", "type": "moneyline"}]},
                {"book": book, "legs": [{"team": "Knicks", "type": "moneyline"}]},   # repeated by overlap
                {"book": book, "legs": [{"team": "Giants", "type": "spread", "line": 3.5}]}]

    monkeypatch.setattr(slip_reader, "read_slip_text", one)
    monkeypatch.setattr(slip_reader, "read_slips_text", many)
    return calls


def test_screenshots_are_read_in_parallel_and_keep_their_order(fake_parsers):
    running, peak, lock = [0], [0], threading.Lock()

    def slow_transcribe(b, m):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.2)
        with lock:
            running[0] -= 1
        return f"Team{b.decode()} ML -110"

    files = [(str(i).encode(), "x/raw") for i in range(4)]
    start = time.time()
    slips, errors = read_screenshots(files, "each", transcribe=slow_transcribe)
    assert time.time() - start < 0.6          # 4 x 0.2s done in parallel, not 0.8s
    assert peak[0] >= 2
    assert [s["legs"][0]["team"] for s in slips] == ["Team0", "Team1", "Team2", "Team3"]
    assert errors == []


def test_one_long_slip_uses_a_single_nemotron_call(fake_parsers):
    files = [(b"a", "x/raw"), (b"b", "x/raw"), (b"c", "x/raw")]
    slips, _ = read_screenshots(files, "one", transcribe=lambda b, m: f"part-{b.decode()}")
    assert len(fake_parsers["one"]) == 1 and len(slips) == 1
    assert fake_parsers["one"][0] == "part-a\npart-b\npart-c"     # in order


def test_my_bets_list_dedupes_slips_repeated_across_screenshots(fake_parsers):
    slips, _ = read_screenshots([(b"a", "x/raw"), (b"b", "x/raw")], "many", book="Hard Rock Bet",
                                transcribe=lambda b, m: "Knicks ML -150")
    assert len(fake_parsers["many"]) == 1
    assert [s["legs"][0]["team"] for s in slips] == ["Knicks", "Giants"]
    assert all(s["book"] == "Hard Rock Bet" for s in slips)


def test_one_bad_screenshot_does_not_sink_the_rest(fake_parsers):
    def flaky(b, m):
        if b == b"bad":
            raise RuntimeError("vision model timed out")
        return "Knicks ML -150"

    slips, errors = read_screenshots([(b"ok", "x/raw"), (b"bad", "x/raw")], "each", transcribe=flaky)
    assert len(slips) == 1
    assert errors == ["Screenshot 2: vision model timed out"]


def test_tiles_of_a_tall_screenshot_are_joined_before_parsing(fake_parsers):
    seen = []
    read_screenshots([(_png(1080, 9000), "image/png")], "many",
                     transcribe=lambda b, m: seen.append(len(b)) or "tile text")
    assert len(seen) >= 4                               # every tile transcribed
    assert fake_parsers["many"][0].count("tile text") == len(seen)
