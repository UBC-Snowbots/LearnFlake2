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

`--source` takes a camera index (`0`), a `/dev/video*` path, a stream URL (`rtsp://…`, `udp://…`), or a GStreamer pipeline ending in `appsink`; these are live sources, timestamped by the capture clock. Anything else is a local video path. Use `--max-frames 150` for a bounded unattended smoke. Use `--help` for available options. The demo is headless and saves an overlay video and machine-readable output to the chosen directory. Stop camera capture with Ctrl-C. Timing is for this experimental pipeline, not a robotics deadline guarantee; do not run it on a loaded rover without operator supervision.

To test the Jetson with a camera plugged into a laptop, stream H.264 over the USB link and decode it on the Jetson's hardware decoder. RTP/MJPEG from ffmpeg decodes with corrupted colour, so don't use it:

```sh
# laptop (Windows, RealSense D435i RGB)
ffmpeg -f dshow -video_size 640x480 -framerate 30 -i "video=Intel(R) RealSense(TM) Depth Camera 435i RGB" -an \
  -c:v libx264 -preset ultrafast -tune zerolatency -g 15 -bf 0 -pix_fmt yuv420p -b:v 6M -f mpegts "udp://192.168.55.1:5000?pkt_size=1316"
# Jetson
python3 -m orion_bench.live_demo ... --device cuda --source "udpsrc port=5000 caps=video/mpegts ! tsdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
```

The temporal model needs a history of 48 pose samples at 15 Hz (roughly 3.2 s); scores are not immediate. It expects a consistent tracked person, image-normalized COCO-17 keypoints and confidence. Occlusion, tracking identity changes, camera motion, target distance, and missed poses can invalidate the history. The validation threshold was calibrated on a different pose source and dataset, not this camera pipeline; it must not be interpreted as a calibrated probability of intent.

## ROS 2 camera input

On the rover, Orion takes its frames from RoverFlake2's ROS 2 camera topics instead of opening the camera itself:

```sh
source /opt/ros/jazzy/setup.bash
python3 -m orion_bench.live_demo \
  --pose models/yolo11n_pose_640 --classifier models/attention_gcn_w48_s0 \
  --source ros2 --device cuda --output-dir results/ros
```

`--source ros2` reads `/realsense435i/image_raw_decoded`. `--source ros2:<topic>` reads any `sensor_msgs/msg/Image` topic, for example `ros2:/logitech/image_raw_decoded`, `ros2:/realsense435i/image_raw` or `ros2:/camera/color/image_raw`. It is a live source like a camera: overlay video, `frames.jsonl` and `summary.json` behave the same.

**Where the frames come from (RoverFlake2 main, package `cameras_cpp`).** This is the pipeline Orion is built around.

1. `jetson_compressor.launch.py` runs one `v4l2_camera_node` per camera in `config/cameras.json` (`realsense415`, `realsense435i`, `logitech`, `suyin`, `science_webcam`, `science_fisheye`), 640x480 by default, on `/{name}/image_raw`. Its `ffmpeg_image_transport` encoder (`hevc_nvmpi`, 300 kbit/s, GOP 7, reliable, depth 5) publishes `/{name}/image_raw/ffmpeg`. This launch file runs on the rover.
2. `jetson_decompressor.launch.py` runs `image_transport republish ffmpeg raw`, turning `/{name}/image_raw/ffmpeg` into plain images on `/{name}/image_raw_decoded`. This launch file runs on the base station.

The package README names these two launch files the other way round (it says to run the encoder with `jetson_decompressor.launch.py`); the launch file contents above are what they actually do. The README also says to use the decoded topic only.

**Decoded or raw.** `/{name}/image_raw_decoded` exists wherever the decoder runs, normally the base station. For Orion on the rover Jetson you have two options:

- Run `jetson_decompressor.launch.py` on the Jetson as well, so the decoded topic exists locally (it only needs the local `/{name}/image_raw/ffmpeg`). It keeps the default source name.
- Use the local raw topic, `--source ros2:/{name}/image_raw`. It avoids both the HEVC quality loss at 300 kbit/s and the extra decode work. Keep every subscriber of the raw topic on the Jetson (about 27 MB/s per 640x480 RGB camera at 30 Hz); do not subscribe to it from the base station over the radio link.

Pose keypoints, and therefore the classifier, should be better on the raw image, so raw is the likely better choice on the Jetson, but neither path has been measured and the default stays on the decoded topic. Compare the two before deciding.

**Image encodings.** Frames are converted to BGR without cv_bridge. Supported: `rgb8`, `bgr8`, `mono8`, `rgba8`, `bgra8`, `yuv422_yuy2` (`yuyv`) and `yuv422` (`uyvy`, 4:2:2 converted with BT.601 limited-range coefficients). Anything else (`16UC1`, `32FC1`, Bayer, ...) stops the run with an error naming the encoding. The image size must stay constant. The decoder republishes the camera's original encoding, and `v4l2_camera` outputs `rgb8` unless its `output_encoding` parameter says otherwise. Older `ffmpeg_image_transport` releases output `bgr8`. `summary.json` records what was seen in `ros_image_encoding`.

**Timestamps.** `timestamp_seconds` is the image header stamp minus the first frame's stamp, so it starts near 0 like the other live sources. `summary.json` records `timestamp_basis: ros_header_stamp` and `ros_stamp_origin_seconds`; each `frames.jsonl` record also has `ros_stamp_seconds`. A zero stamp or a stamp that goes backwards stops the run, because the tracker's windows depend on time order.

**Latency, QoS and staleness.**

- Only the newest frame is kept (older ones are dropped, like `appsink drop=true`). If Orion is slower than the camera it processes fewer frames per second; it does not fall behind.
- The subscription is best effort with history depth 1. A best-effort subscriber receives from both best-effort and reliable publishers. A reliable subscriber (`--ros-reliable`) only receives from reliable ones, so leave it off unless you need it. The unmerged `cameras.cpp` publishes reliable, depth 10.
- With no new frame for `--ros-warn-after` seconds (default 2) a warning is printed, naming the topic and how many publishers are visible. Zero publishers means a wrong topic name, a node that is not running, a different `ROS_DOMAIN_ID` or no network route. After `--ros-timeout` seconds (default 10) the run fails with exit code 2 and `status: error` in `summary.json`.
- Ctrl-C or SIGTERM stops the ROS node and writes the summary with `status: interrupted`.

**Python environment on the Jetson.** The ROS 2 Jazzy Python packages are built for the system Python 3.12 of Ubuntu 24.04, and they have to be visible to the interpreter that runs Orion. Either use the system `python3` (with PyTorch and OpenCV installed for it), or build the venv on that interpreter with access to system packages, after sourcing ROS:

```sh
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages ~/orion-env
source ~/orion-env/bin/activate        # then source /opt/ros/jazzy/setup.bash again in each new shell
python3 -c 'import rclpy, sensor_msgs; print("ROS OK")'
```

Without ROS on the path the demo exits with a message saying rclpy cannot be imported; nothing else in Orion needs ROS. Whether the Jetson has ROS 2 Jazzy installed has not been checked.

**Testing without a camera.** From an Orion checkout (the script is not in the kit) publish synthetic or recorded frames in a second terminal, with ROS sourced:

```sh
python3 scripts/publish_test_images.py                       # 640x480 rgb8, 30 Hz, /realsense435i/image_raw_decoded
python3 scripts/publish_test_images.py --video clip.mp4 --encoding yuv422_yuy2 --topic /logitech/image_raw_decoded
```

Unit tests for the conversions and the frame buffer: `python3 -m pytest tests -q` from the Orion directory (the rclpy test skips when ROS is not available). The path was run end to end on the workstation with ROS 2 Jazzy, CPU and synthetic frames; it has not been run on the Jetson or with a real camera.

**The other publisher: `cameras.cpp` (unmerged).** Branch `cameron/cameras_cpp` has `rover_vision_cpp/src/cameras.cpp` (empty on main), node `camera_pub_node`, publishing `sensor_msgs/Image` at about 30 Hz, 640x480, `header.stamp` = now, reliable depth 10. RealSense colour is `/camera/color/image_raw` (`rgb8`), infrared `/camera/infra1/image_rect_raw` and `/camera/infra2/image_rect_raw` (`mono8`), depth `/camera/depth/image_rect_raw` (`16UC1`); each standard webcam is `cam_0`, `cam_1`, ... (`rgb8`). Use `--source ros2:/camera/color/image_raw` for it. Orion does not use depth, and the depth image is **not aligned to the colour image**: before any range use the publisher would need `rs2::align` (or the colour and depth would have to be reprojected with the camera intrinsics and extrinsics).

## Repack from Orion

From an Orion checkout with both prepared model artifacts present:

```sh
python3 scripts/package_live_kit.py --output /absolute/path/to/orion-live-kit.tar.gz
```

The source checkpoint/pose model is Ultralytics YOLO11n-pose, offered under AGPL-3.0 or an Enterprise license; review obligations before distributing the kit or integrating it with closed-source software. The classifier was trained using NTU RGB+D 120 research data and HMDB51; review applicable terms before redistribution or non-research use. Source and artifact hashes are in the bundled model manifests.
