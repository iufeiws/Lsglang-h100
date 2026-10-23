#!/usr/bin/env python3
"""Seed the sgl_eval gsm8k cache from a locally-obtained, byte-verified copy.

WHY THIS EXISTS
---------------
`sgl_eval`'s gsm8k spec uses ``"loader": "prepare"`` (sgl_eval/evals/_registry.py:38-41),
not ``"bundled"``.  The vendored prepare script
(``sgl_eval/_vendored/nemo_skills/dataset/gsm8k/prepare.py:23``) downloads

    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl

which is **unreachable from this host** (measured: ``http=000``).  The script also
writes its output *beside its own* ``__file__``, which sgl_eval redirects into

    ~/.cache/sgl_eval/gsm8k/test.jsonl            (+ marker test.jsonl.sha256)

so seeding that cache with a faithful copy bypasses the network entirely.

``_cache_is_valid()`` (sgl_eval/evals/_loader.py:218-224) requires BOTH the file
and a marker holding ``sha256(file).hexdigest() + "\\n"``.  Dropping only the
jsonl is NOT enough -- sgl_eval would re-run prepare and fail on the network.

SOURCE DATASET (verified byte-identical to canonical openai/gsm8k)
------------------------------------------------------------------
ModelScope ``AI-ModelScope/gsm8k`` -> ``main/test-00000-of-00001.parquet``
    size   419088 bytes
    sha256 ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59
which equals exactly the git-lfs pointer that hf-mirror.com serves for
``datasets/openai/gsm8k/main/test-00000-of-00001.parquet`` (the LFS payload itself
is behind an unreachable cas-bridge host, which is why we take the ModelScope copy).

Because the source is byte-identical to canonical, the resulting test.jsonl -- and
therefore the score -- stays comparable with the repository's 0.95 threshold.

This script reimplements the SAME transformation the vendored prepare.py applies,
so its output is what the network path would have produced.  Run with ``--check``
to verify an existing cache instead of writing one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Copied verbatim from sgl_eval/_vendored/nemo_skills/dataset/gsm8k/prepare.py
# (the 11 known-bad training answers the vendored script corrects).  Kept here so
# the seeded file is identical to what the network path would have written.
# ---------------------------------------------------------------------------
FIXES = {
    """Mr. Finnegan has 3 tanks with a capacity of 7000 gallons, 5000 gallons, and 3000 gallons, respectively. If he fills the first tank up to 3/4 full, the second tank with water up to 4/5 of its capacity, and the third tank up to half of its capacity, how many gallons in total are in the tanks?""": 10750,
    """If you double a number and add 5 to the result, then that's 20 more than half of the original number. What's the original number?""": 10,
    """John has 2 hives of bees.  One of the hives has 1000 bees and produces 500 liters of honey.  The second has 20% fewer bees but each bee produces 40% more honey.  How much honey does he produce?""": 2740,
    """Three blue chips are in a jar which is 10% of the entire chips. If 50% of the chips are white and the rest are green, how many green chips are there?""": 15,
    """Janet filmed a new movie that is 60% longer than her previous 2-hour long movie.  Her previous movie cost $50 per minute to film, and the newest movie cost twice as much per minute to film as the previous movie.  What was the total amount of money required to film Janet's entire newest film?""": 19200,
    """Students at Highridge High earn 2 points for each correct answer during a quiz bowl If a student correctly answers all the questions in a round, the student is awarded an additional 4 point bonus. They played a total of five rounds each consisting of five questions. If James only missed one question, how many points did he get?""": 64,
    """Robert and Teddy are planning to buy snacks for their friends.  Robert orders five boxes of pizza at $10 each box and ten cans of soft drinks at $2 each. Teddy buys six hamburgers at $3 each and an additional ten cans of soft drinks. How much do they spend in all?""": 108,
    """James invests $2000 a week into his bank account.  He had $250,000 in his account when the year started.  At the end of the year, he gets a windfall that is worth 50% more than what he has in his bank account.   How much money does he have?""": 531000,
    """James does chores around the class.  There are 3 bedrooms, 1 living room, and 2 bathrooms to clean.  The bedrooms each take 20 minutes to clean.  The living room takes as long as the 3 bedrooms combined.  The bathroom takes twice as long as the living room.  He also cleans the outside which takes twice as long as cleaning the house.  He splits the chores with his 2 siblings who are just as fast as him.  How long, in hours, does he work?""": 6,
    """During one game, a total of 50 people attended a baseball team’s games. Forty percent and thirty-four percent of the audiences are supporters of the first and second teams, respectively. How many people attended the game did not support either of the teams?""": 13,
    """Pete has to take a 10-minute walk down to the train station and then board a 1hr 20-minute train to LA. When should he leave if he cannot get to LA later than 0900 hours? (24-hr time)""": "07:30",
}

REPLACE_MARKER = re.compile(r"<<.*?>>")


def transform(rows):
    """Row-wise transform of prepare.py::save_data, applied to (question, answer)."""
    out = []
    n_fixed = 0
    for question, answer in rows:
        entry = {"problem": question}
        solution, expected_answer = answer.split("####")
        value = float(expected_answer.replace(",", ""))
        if int(value) == value:
            value = int(value)
        entry["expected_answer"] = value
        entry["reference_solution"] = REPLACE_MARKER.sub("", solution)
        if question in FIXES:
            entry["expected_answer"] = FIXES[question]
            n_fixed += 1
        out.append(entry)
    return out, n_fixed


def read_parquet(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit(
            "pyarrow is required to read the parquet source.\n"
            f"  target env: /mnt/SSD01/models/.llm-env/Lsglang/env/bin/python -c 'import pyarrow'\n"
            f"  or pass --from-jsonl <original test.jsonl> instead."
        )
    table = pq.read_table(path, columns=["question", "answer"])
    return list(zip(table.column("question").to_pylist(), table.column("answer").to_pylist()))


def read_original_jsonl(path: Path):
    rows = []
    with path.open("rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows.append((row["question"], row["answer"]))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_cache(entries, dest_dir: Path) -> Path:
    """Write test.jsonl + test.jsonl.sha256 atomically (staged dir then rename).

    The marker text is exactly ``sha256 + "\\n"`` -- see
    sgl_eval/evals/_loader.py::_commit_with_marker / ::_cache_is_valid.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".staging-seed-", dir=str(dest_dir)))
    staged = tmp_dir / "test.jsonl"
    with staged.open("wt", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    digest = sha256_file(staged)
    (tmp_dir / "test.jsonl.sha256").write_text(digest + "\n")
    final = dest_dir / "test.jsonl"
    os.replace(staged, final)
    os.replace(tmp_dir / "test.jsonl.sha256", Path(str(final) + ".sha256"))
    tmp_dir.rmdir()
    return final


def verify(cache_dir: Path, num_examples: int) -> int:
    """Reproduce _cache_is_valid() + report the deterministic first-N selection."""
    data = cache_dir / "test.jsonl"
    marker = Path(str(data) + ".sha256")
    if not (data.is_file() and marker.is_file()):
        print(f"FAIL: missing {data} or {marker}")
        return 1
    actual = sha256_file(data)
    stored = marker.read_text().strip()
    if actual != stored:
        print(f"FAIL: marker mismatch\n  file   {actual}\n  marker {stored}")
        return 1
    rows = [json.loads(l) for l in data.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"OK: cache valid at {data}")
    print(f"    rows={len(rows)}  sha256={actual}")
    print(f"    marker matches: yes")
    n = min(num_examples, len(rows))
    print(f"    gsm8k has no sample_seed -> _select_rows takes the FIRST {n} rows in file order")
    print(f"    row[0].problem = {rows[0]['problem'][:90]!r}")
    print(f"    row[0].expected_answer = {rows[0]['expected_answer']!r}")
    if rows and "reference_solution" in rows[0]:
        print(f"    row[0].reference_solution head = {rows[0]['reference_solution'][:60]!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-parquet", type=Path, help="canonical openai/gsm8k main/test parquet")
    src.add_argument("--from-jsonl", type=Path, help="original grade-school-math test.jsonl")
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path.home() / ".cache" / "sgl_eval" / "gsm8k",
        help="cache dir to write (default: the dir sgl_eval reads)",
    )
    ap.add_argument("--build-artifact", type=Path,
                    help="write the prepared test.jsonl into this dir INSTEAD of installing the cache")
    ap.add_argument("--check", action="store_true", help="verify an existing cache and exit")
    ap.add_argument("--num-examples", type=int, default=200)
    ap.add_argument("--verify-sha256", type=str,
                    help="assert the source file sha256 before using it")
    args = ap.parse_args()

    if args.check:
        return verify(args.dest, args.num_examples)

    if args.from_parquet:
        source = args.from_parquet
        if args.verify_sha256:
            got = sha256_file(source)
            if got != args.verify_sha256:
                sys.exit(f"source sha256 mismatch\n  got      {got}\n  expected {args.verify_sha256}")
        rows = read_parquet(source)
        origin = f"parquet {source}"
    elif args.from_jsonl:
        source = args.from_jsonl
        if args.verify_sha256:
            got = sha256_file(source)
            if got != args.verify_sha256:
                sys.exit(f"source sha256 mismatch\n  got      {got}\n  expected {args.verify_sha256}")
        rows = read_original_jsonl(source)
        origin = f"jsonl {source}"
    else:
        sys.exit("give --from-parquet or --from-jsonl (or --check)")

    entries, n_fixed = transform(rows)
    print(f"source      : {origin}")
    print(f"questions   : {len(rows)}")
    print(f"fixes applied: {n_fixed} (prepare.py corrects known-bad answers)")

    dest = args.build_artifact or args.dest
    out = write_cache(entries, dest)
    print(f"wrote       : {out}")
    print(f"marker      : {out}.sha256 -> {sha256_file(out)}")
    if args.build_artifact:
        print("NOTE: artifact only. Install into the sgl_eval cache with:")
        print(f"      cp {out} {out}.sha256 <cache_dir>/   (cache_dir = ~/.cache/sgl_eval/gsm8k)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
