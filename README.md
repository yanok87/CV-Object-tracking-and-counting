# Object Tracking — Drone Counting

Learning project that detects and tracks drones in video with **YOLO**, counts them when they cross a line, and draws an ROI with **OpenCV**.

Adapted from apple object counting to a **drones** use case.

## Credits

This was a learning project based on:

- Video: [How to Build Real-Time Apple Object Counting with YOLO Tracking + OpenCV](https://youtu.be/ybCzLHJdpbI?si=WqWfJrikrYoqDORU)
- Repo: [NitinCVOrbit/How-to-Build-Real-Time-Apple-Object-Counting-with-YOLO-Tracking-OpenCV-Computer-Vision](https://github.com/NitinCVOrbit/How-to-Build-Real-Time-Apple-Object-Counting-with-YOLO-Tracking-OpenCV-Computer-Vision/tree/main)

## What it does

1. Loads a YOLO model (`best.pt`)
2. Reads a video from `input_video/`
3. Masks a polygon ROI and runs detection + tracking inside it
4. Smooths object centers across frames
5. Counts unique track IDs when their center crosses a counting line
6. Shows a live preview and saves the result to `output_video/`

## Project layout

| Path | Purpose |
|------|---------|
| `main.py` | Main tracking + counting app |
| `pick_roi.py` | Click on a frame to choose ROI and line points |
| `help.py` | Reference-only explanations of the math/logic (not used at runtime) |
| `best.pt` | Trained YOLO weights |
| `input_video/` | Input videos (e.g. `drones_swarm.mp4`) |
| `output_video/` | Written result videos |
| `model/` | Dataset + training scripts / notebook |

## Setup

Requires Python 3.10+ (tested with 3.14 locally).

```bash
cd ObjectTracking
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install ultralytics opencv-python numpy
```

### Input video

The sample `drones_swarm.mp4` was taken from [Shutterstock](https://www.shutterstock.com/). You can use your own video instead: put it in `input_video/` and update the path in `main.py` (and `pick_roi.py` if you re-pick ROI points).

Default path:

```text
input_video/drones_swarm.mp4
```

Also ensure `best.pt` is in the project root (or update the path in `main.py`).

## Configure ROI and counting line

Points must match the resized frame size used in `main.py` (**1280×720**).

```bash
python pick_roi.py
```

1. Click **4** ROI corners (in order around the polygon)
2. Click **2** line endpoints
3. Press any key and paste the printed lists into `main.py`:

```python
roi_points = [...]
line_points = [...]
```

## Run

From the project root (with the venv active):

```bash
python main.py
```

- Press **q** to quit early
- Output is saved to `output_video/drones_swarm_counted.mp4`

## Optional: train your own model

Training scripts live under `model/` (local script and Colab-friendly notebook). For Colab, use a **GPU** runtime (e.g. T4) — CPU training is very slow.

```bash
python model/train_model_drones_dataset.py
```

After training, copy the best weights to the project root as `best.pt` (or point `main.py` at the new path).
