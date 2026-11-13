"""Turn T-DEED events into H/A/D/X labels for every frame.

Run from the main folder of the repo (after run_tdeed.py):
    python solution/tdeed/events_to_labels.py --name tdeed_v1

Writes evaluation/results/<name>.csv. Then:
    python evaluation/validate_submission.py evaluation/results/<name>.csv
    python evaluation/run_evaluation.py

The rules, kept simple on purpose (change them in to_labels()):
  1. A pass, drive, shot, cross, header, throw-in or free kick means that team has the ball.
     T-DEED does not say which team, so the team is read from the tracking: the team nearest the
     ball in the half-second around the event.
  2. OUT or GOAL means a dead ball (D) until the next event that restarts play.
  3. Between two events of the same team, that team keeps the ball. Between events of different
     teams, the change happens just after the first team was last seen nearest the ball.
  4. Before the first event, and where the team cannot be read, the label is X (abstain).
Numbers like MIN_CONFIDENCE are plain defaults: do not tune them on the public 5 minutes.
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "starter"))

EVENTS_CSV = os.path.join(ROOT, "data", "derived", "tdeed_events.csv")
CODES_CACHE = os.path.join(ROOT, "data", "cache_nearest_codes.npy")
N_FRAMES = 199319

HOME, AWAY, NOBODY = 0, 1, -1
POSSESSION = {"PASS", "HIGH PASS", "CROSS", "DRIVE", "SHOT", "HEADER", "THROW IN", "FREE KICK"}
STOPS = {"OUT", "GOAL"}
MIN_CONFIDENCE = 0.3
TEAM_WINDOW = 12  # frames each side of an event (about 0.5 s) used to read the team


def load_events(path=EVENTS_CSV):
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return [(int(r["frame"]), r["label"], float(r["confidence"])) for r in rows]


def load_codes():
    """Nearest-team code per frame (0 home, 1 away, -1 ball not seen). Cached after the first run."""
    if os.path.exists(CODES_CACHE):
        return np.load(CODES_CACHE)
    import baseline  # the organisers' starter/baseline.py
    codes = baseline.build()
    np.save(CODES_CACHE, codes)
    return codes


def team_at(codes, frame):
    window = codes[max(0, frame - TEAM_WINDOW): frame + TEAM_WINDOW + 1]
    home, away = int(np.sum(window == HOME)), int(np.sum(window == AWAY))
    if home == away:
        return NOBODY
    return HOME if home > away else AWAY


def to_labels(events, codes, n_frames=N_FRAMES):
    letter = {HOME: "H", AWAY: "A"}
    labels = np.full(n_frames, "X", dtype="<U1")

    # Keep confident events, each tagged with a team (possession) or 'D' (stop).
    marks = []
    for frame, label, confidence in events:
        if confidence < MIN_CONFIDENCE or not 0 <= frame < n_frames:
            continue
        if label in STOPS:
            marks.append((frame, "D"))
        elif label in POSSESSION:
            team = team_at(codes, frame)
            if team != NOBODY:
                marks.append((frame, letter[team]))
    marks.sort()

    for (f1, s1), (f2, s2) in zip(marks, marks[1:] + [(n_frames, None)]):
        if s1 == "D" or s2 is None or s1 == s2:
            labels[f1:f2] = s1                      # same state until the next event
            continue
        if s2 == "D":
            labels[f1:f2] = s1                      # play goes on until the stop
            continue
        # Change of team between f1 and f2: switch right after the first team was last nearest.
        seen = np.nonzero(codes[f1:f2] == (HOME if s1 == "H" else AWAY))[0]
        switch = f1 + int(seen[-1]) + 1 if len(seen) else (f1 + f2) // 2
        labels[f1:switch] = s1
        labels[switch:f2] = s2
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="output file name, e.g. tdeed_v1")
    args = parser.parse_args()

    import baseline
    labels = to_labels(load_events(), load_codes())
    out = os.path.join(ROOT, "evaluation", "results", args.name + ".csv")
    baseline.write_submission(out, labels)
    print("wrote", os.path.relpath(out, ROOT), {k: int(np.sum(labels == k)) for k in "HADX"})


if __name__ == "__main__":
    main()
