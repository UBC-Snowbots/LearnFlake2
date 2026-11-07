# Orion portable video test kit

This kit is for **offline experiments**, not rover control. It combines YOLO11n-pose with an experimental temporal classifier trained on pre-extracted HRNet poses. The models have not been validated together, on Jetson, on beckoning or on leg waving. Its score is not a reliable attention/intent decision; false alarms and missed people are expected. Do not connect it to autonomous motion or safety systems.

## Transfer and integrity

Extract on the target computer, then verify the included files before using them:

```sh
tar -xzf orion-live-kit.tar.gz
cd orion-live-kit
sha256sum -c SHA256SUMS
```

The archive includes trained model artifacts, their provenance manifests and Python source; it does **not** include Python, PyTorch, OpenCV, CUDA libraries, input videos, datasets or a TensorRT engine. Transfer it and dependencies by USB if the target is offline. Treat model artifacts as code: load only archives from a trusted source and verify the checksum obtained through a trusted channel.

## Install prerequisites

Linux x86-64 desktop or NVIDIA Jetson with Python 3 and a compatible PyTorch build, NumPy and OpenCV (`cv2`, including video I/O). Run commands from the extracted kit root. On desktop, create a Python environment and install packages appropriate to your OS and GPU; on Jetson install NVIDIA's PyTorch wheel that matches the installed JetPack, CUDA, Python and architecture: [NVIDIA Jetson PyTorch installation](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html). Obtain wheels and dependencies beforehand if the target has no internet. Avoid replacing the robotics stack's system Python: use an isolated environment. Jetson OpenCV may be supplied by the OS; expose it to that environment or install a compatible build.

```sh
python3 -c 'import torch, cv2, numpy; print("torch", torch.__version__, "CUDA runtime", torch.version.cuda, "CUDA available", torch.cuda.is_available()); print("OpenCV", cv2.__version__, "NumPy", numpy.__version__)'
```

CUDA mode requires `torch.cuda.is_available()` to be true; there is no silent CPU fallback. A Jetson PyTorch wheel/version may not be compatible with these TorchScript files, exported with PyTorch 2.14.1+cu130. If loading fails, use a compatible PyTorch environment or re-export the models for the target; successful Jetson installation and execution have not been verified. TensorRT and ONNX are not used by this demo.

## Run a recorded video or camera

```sh
python3 -m orion_bench.live_demo \
  --pose models/yolo11n_pose_640 \
  --classifier models/attention_gcn_w48_s0 \
  --source /absolute/path/to/video.mp4 --device cpu --output-dir results/video

# NVIDIA GPU, once CUDA-enabled PyTorch is installed:
python3 -m orion_bench.live_demo \
  --pose models/yolo11n_pose_640 \
  --classifier models/attention_gcn_w48_s0 \
  --source 0 --device cuda --output-dir results/camera
```

A numeric source is an OpenCV camera index; a nonnumeric source is a video path. Use `--max-frames 150` for a bounded unattended smoke. Use `--help` for available options. The demo is headless and saves an overlay video and machine-readable output to the chosen directory. Stop camera capture with Ctrl-C. Timing is for this experimental pipeline, not a robotics deadline guarantee; do not run it on a loaded rover without operator supervision.

The temporal model needs a history of 48 pose samples at 15 Hz (roughly 3.2 s); scores are not immediate. It expects a consistent tracked person, image-normalized COCO-17 keypoints and confidence. Occlusion, tracking identity changes, camera motion, target distance, and missed poses can invalidate the history. The validation threshold was calibrated on a different pose source and dataset, not this camera pipeline; it must not be interpreted as a calibrated probability of intent.

## Repack from Orion

From an Orion checkout with both prepared model artifacts present:

```sh
python3 scripts/package_live_kit.py --output /absolute/path/to/orion-live-kit.tar.gz
```

The source checkpoint/pose model is Ultralytics YOLO11n-pose, offered under AGPL-3.0 or an Enterprise license; review obligations before distributing the kit or integrating it with closed-source software. The classifier was trained using NTU RGB+D 120 research data and HMDB51; review applicable terms before redistribution or non-research use. Source and artifact hashes are in the bundled model manifests.
