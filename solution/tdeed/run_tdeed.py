"""Run the pretrained T-DEED ball-action model on the tactical video and save its events.

Run from the main folder of the repo:
    python solution/tdeed/run_tdeed.py

Output: data/derived/tdeed_events.csv with columns frame,label,confidence
(frame = frame of the tactical video = frame of the submission file).

Needs an NVIDIA GPU. The model is used as downloaded: it is never trained on this match.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TDEED = os.path.join(ROOT, "third_party", "T-DEED")
VIDEO = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical.mp4")
OUT_CSV = os.path.join(ROOT, "data", "derived", "tdeed_events.csv")  # data/ is git-ignored


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="SoccerNetBall_challenge2",
                        help="T-DEED config name; its checkpoint must be downloaded first")
    parser.add_argument("--video", default=VIDEO)
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="keep events above this confidence (T-DEED's own default)")
    args = parser.parse_args()

    checkpoint = os.path.join(TDEED, "checkpoints", "SoccerNetBall", args.model, "checkpoint_best.pt")
    if not os.path.isdir(TDEED):
        sys.exit("T-DEED is missing. Run: git submodule update --init")
    if not os.path.exists(checkpoint):
        sys.exit("Checkpoint missing. Download the '{}' folder from the T-DEED README link and put "
                 "checkpoint_best.pt here:\n  {}".format(args.model, checkpoint))
    if not os.path.exists(args.video):
        sys.exit("Video missing: {}\nUnzip the data bundle into data/.".format(args.video))

    # T-DEED's own inference script: reads the video, resizes to 796x448 (its training size),
    # looks at every 2nd frame, writes inference_output/results_inference.json inside T-DEED.
    subprocess.run([sys.executable, "inference.py",
                    "--model", args.model,
                    "--video_path", os.path.abspath(args.video),
                    "--frame_width", "796", "--frame_height", "448",
                    "--inference_threshold", str(args.threshold)],
                   cwd=TDEED, check=True)

    with open(os.path.join(TDEED, "inference_output", "results_inference.json")) as handle:
        events = json.load(handle)["predictions"]
    events.sort(key=lambda e: e["frame"])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "label", "confidence"])
        for e in events:
            writer.writerow([e["frame"], e["label"], round(e["confidence"], 4)])

    print("wrote {} events to {}".format(len(events), os.path.relpath(OUT_CSV, ROOT)))
    for label, n in Counter(e["label"] for e in events).most_common():
        print("  {:<26} {}".format(label, n))


if __name__ == "__main__":
    main()
