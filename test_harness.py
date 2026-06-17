#!/usr/bin/env python3
"""
SwingIQ accuracy & consistency test harness.

Runs a folder of test videos through the /analyze endpoint multiple times and
logs the results to a CSV so you can check:

  1. Frame-selection accuracy  — which frame number was chosen for each swing
     position (compare against a ground-truth file you fill in by hand).
  2. Run-to-run consistency     — does the same video produce the same major
     findings each time? (A reliable model should.)

USAGE
-----
  # Against your local dev server (python app.py running on :5000)
  python test_harness.py --videos ./test_videos --runs 3

  # Against the live Railway site, using your own API key so you don't burn
  # the free-tier limit:
  python test_harness.py --url https://swingiq.up.railway.app \
      --videos ./test_videos --runs 3 --api-key sk-ant-...

  # Tag every video with the club used:
  python test_harness.py --videos ./test_videos --club "7-Iron"

OUTPUT
------
  results.csv          — one row per run (frames chosen, issue counts, issues)
  A consistency report printed to the console.

Put your test videos in a folder. Supported: .mp4 .mov .avi .m4v
"""

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict

try:
    import httpx  # already installed (anthropic depends on it)
except ImportError:
    sys.exit("httpx not found. Run:  pip install httpx")


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".m4v"}


def extract_issues(analysis_text: str):
    """
    Pull issue titles out of the markdown report, grouped by severity.
    Looks under '### Major Issues', '### Moderate Issues', '### Minor Issues'.
    Returns dict: {"major": [...], "moderate": [...], "minor": [...]}
    """
    sections = {"major": [], "moderate": [], "minor": []}
    current = None
    for line in analysis_text.splitlines():
        low = line.lower().strip()
        if low.startswith("###"):
            if "major" in low:
                current = "major"
            elif "moderate" in low:
                current = "moderate"
            elif "minor" in low:
                current = "minor"
            else:
                current = None
            continue
        if current and line.strip():
            # Grab a bold title or a bullet as the issue label
            m = re.match(r"^[-*]\s*\*\*(.+?)\*\*", line.strip()) \
                or re.match(r"^\*\*(.+?)\*\*", line.strip()) \
                or re.match(r"^[-*]\s+(.+)", line.strip())
            if m:
                title = m.group(1).strip().rstrip(":").lower()
                # keep it short for comparison
                sections[current].append(title[:60])
    return sections


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def run_one(client, url, video_path, club, api_key):
    """POST a single video to /analyze. Returns (elapsed, json) or raises."""
    with open(video_path, "rb") as f:
        files = {"front_video": (os.path.basename(video_path), f, "video/mp4")}
        data = {}
        if club:
            data["club"] = club
        if api_key:
            data["api_key"] = api_key
        t0 = time.time()
        resp = client.post(f"{url}/analyze", files=files, data=data, timeout=600)
        elapsed = time.time() - t0
    resp.raise_for_status()
    return elapsed, resp.json()


def main():
    ap = argparse.ArgumentParser(description="SwingIQ accuracy test harness")
    ap.add_argument("--url", default="http://localhost:5000", help="Base URL of the server")
    ap.add_argument("--videos", required=True, help="Folder of test videos")
    ap.add_argument("--runs", type=int, default=3, help="Times to run each video")
    ap.add_argument("--club", default="", help="Club to send for every video")
    ap.add_argument("--api-key", default="", help="Anthropic key (avoids free-tier limit)")
    ap.add_argument("--out", default="results.csv", help="CSV output path")
    args = ap.parse_args()

    videos = sorted(
        os.path.join(args.videos, f)
        for f in os.listdir(args.videos)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    )
    if not videos:
        sys.exit(f"No videos found in {args.videos}")

    print(f"Found {len(videos)} videos, {args.runs} run(s) each "
          f"= {len(videos) * args.runs} total analyses against {args.url}\n")

    client = httpx.Client()
    rows = []
    # per-video list of major-issue sets, for consistency scoring
    major_by_video = defaultdict(list)

    for vid in videos:
        name = os.path.basename(vid)
        for run in range(1, args.runs + 1):
            print(f"[{name}] run {run}/{args.runs} ... ", end="", flush=True)
            try:
                elapsed, data = run_one(client, args.url, vid, args.club, args.api_key)
            except Exception as e:
                print(f"FAILED: {e}")
                rows.append({
                    "video": name, "run": run, "status": "ERROR",
                    "elapsed_sec": "", "frames": "", "frame_numbers": "",
                    "major_count": "", "moderate_count": "", "minor_count": "",
                    "major_issues": str(e)[:200],
                })
                continue

            frames = data.get("frames", [])
            # frame numbers for the FRONT view (or whatever is first)
            frame_nums = [f.get("frame") for f in frames if f.get("frame")]
            issues = extract_issues(data.get("analysis", ""))
            major_by_video[name].append(issues["major"])

            print(f"ok ({elapsed:.0f}s) — frames={frame_nums} "
                  f"major={len(issues['major'])} mod={len(issues['moderate'])} min={len(issues['minor'])}")

            rows.append({
                "video": name,
                "run": run,
                "status": "ok",
                "elapsed_sec": f"{elapsed:.1f}",
                "frames": len(frames),
                "frame_numbers": " ".join(str(n) for n in frame_nums),
                "major_count": len(issues["major"]),
                "moderate_count": len(issues["moderate"]),
                "minor_count": len(issues["minor"]),
                "major_issues": " | ".join(issues["major"]),
            })

    # Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "video", "run", "status", "elapsed_sec", "frames", "frame_numbers",
            "major_count", "moderate_count", "minor_count", "major_issues",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # Consistency report
    print("\n=== CONSISTENCY REPORT (major issues, run-to-run) ===")
    print("Higher = more consistent. 1.00 means identical major findings every run.\n")
    for name, runs in major_by_video.items():
        if len(runs) < 2:
            print(f"  {name}: only 1 run — can't score")
            continue
        scores = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                scores.append(jaccard(runs[i], runs[j]))
        avg = sum(scores) / len(scores) if scores else 0
        flag = "  ⚠ INCONSISTENT" if avg < 0.5 else ""
        print(f"  {name}: avg overlap {avg:.2f}{flag}")
    print("\nManual next steps:")
    print("  • Open results.csv and check frame_numbers against your ground truth.")
    print("  • For pro swings, major_count should be low. If high → possible hallucination.")
    print("  • Investigate any video flagged INCONSISTENT above.")


if __name__ == "__main__":
    main()
