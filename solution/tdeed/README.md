# T-DEED: ball events from the video

Pretrained [T-DEED](https://github.com/arturxe2/T-DEED) (SoccerNet Ball Action Spotting) finds passes,
drives, outs, throw-ins, free kicks and goals in the tactical video. We turn those events into H/A/D labels.
The model is used as downloaded and never trained on this match.

## Setup (once)

Run in your terminal, from the main folder of the team repo:

```bash
# 1. Add T-DEED to the repo as a submodule (a link to their code, not a copy)
git submodule add https://github.com/arturxe2/T-DEED third_party/T-DEED

# 2. Install what it needs (an NVIDIA GPU is required)
pip install torch torchvision timm opencv-python tqdm tabulate wandb
```

3. Download the checkpoint: open the "Model checkpoints" link in the
   [T-DEED README](https://github.com/arturxe2/T-DEED#trained-models), open the `SoccerNetBall` folder,
   then `SoccerNetBall_challenge2`, and download `checkpoint_best.pt`. Put it here:

   `third_party/T-DEED/checkpoints/SoccerNetBall/SoccerNetBall_challenge2/checkpoint_best.pt`

4. Make sure `.gitignore` contains these lines, so the data and the checkpoint never reach GitHub:

   ```
   data/
   third_party/T-DEED/checkpoints/
   third_party/T-DEED/inference_output/
   ```

## Run

```bash
python solution/tdeed/run_tdeed.py                      # video -> data/derived/tdeed_events.csv
python solution/tdeed/events_to_labels.py --name tdeed_v1   # events -> evaluation/results/tdeed_v1.csv
python evaluation/validate_submission.py evaluation/results/tdeed_v1.csv
python evaluation/run_evaluation.py
```

The first command takes a while: it reads the whole 2-hour video.

## Where to change things

- **How events become labels:** `to_labels()` in `events_to_labels.py`.
- **Exploring the events:** `tdeed_explore.ipynb`.

## Rules to keep

- Do not train or fine-tune T-DEED on this match's video. Use it as downloaded.
- Do not check the labels against the video by eye.
- Do not tune numbers like `MIN_CONFIDENCE` on the public 5 minutes.
- Do not upload the match video to a cloud service without asking the organisers first.
