"""Latest-frame camera source for a ROS 2 `sensor_msgs/msg/Image` topic.

rclpy is imported lazily and only when a `RosImageSource` is created, so importing this module (and the pure
encoding conversion in `image_to_bgr`) works on machines without ROS. Images are converted without cv_bridge.
Supported encodings: rgb8, bgr8, mono8, rgba8, bgra8, yuv422_yuy2 (yuyv) and yuv422 (uyvy).
"""
import logging
import threading
import time

import numpy as np


log = logging.getLogger("orion.ros_source")

PREFIX = "ros2:"
# Orion's default camera: RoverFlake2 main's cameras_cpp decoder output for the realsense435i entry in cameras.json.
DEFAULT_TOPIC = "/realsense435i/image_raw_decoded"


def yuv422_to_bgr(rows, luma_first):
    """Packed 4:2:2 rows of shape (h, w, 2) to BGR (BT.601 limited range, chroma shared by pixel pairs, like OpenCV)."""
    height, width = rows.shape[:2]
    if width % 2:
        raise ValueError(f"YUV 4:2:2 images need an even width, got {width}")
    quads = rows.reshape(height, width // 2, 4).astype(np.float32)
    first, second = (quads[:, :, 0], quads[:, :, 2]) if luma_first else (quads[:, :, 1], quads[:, :, 3])
    u, v = (quads[:, :, 1], quads[:, :, 3]) if luma_first else (quads[:, :, 0], quads[:, :, 2])
    luma = np.stack((first, second), axis=2).reshape(height, width)
    luma = np.maximum(luma - 16.0, 0.0) * 1.164
    u = np.repeat(u - 128.0, 2, axis=1)
    v = np.repeat(v - 128.0, 2, axis=1)
    bgr = np.empty((height, width, 3), dtype=np.uint8)
    bgr[:, :, 0] = np.clip(luma + 2.017 * u + 0.5, 0, 255)
    bgr[:, :, 1] = np.clip(luma - 0.392 * u - 0.813 * v + 0.5, 0, 255)
    bgr[:, :, 2] = np.clip(luma + 1.596 * v + 0.5, 0, 255)
    return bgr


# encoding -> (bytes per pixel, conversion of an (h, w, bytes per pixel) uint8 view into BGR)
ENCODINGS = {
    "bgr8": (3, lambda a: a),
    "rgb8": (3, lambda a: a[:, :, ::-1]),
    "mono8": (1, lambda a: np.broadcast_to(a, a.shape[:2] + (3,))),
    "bgra8": (4, lambda a: a[:, :, :3]),
    "rgba8": (4, lambda a: a[:, :, 2::-1]),
    "yuv422_yuy2": (2, lambda a: yuv422_to_bgr(a, luma_first=True)),   # Y U Y V
    "yuyv": (2, lambda a: yuv422_to_bgr(a, luma_first=True)),
    "yuv422": (2, lambda a: yuv422_to_bgr(a, luma_first=False)),       # ROS name for U Y V Y
    "uyvy": (2, lambda a: yuv422_to_bgr(a, luma_first=False)),
}


class SourceTimeout(RuntimeError):
    """No new frame arrived within the failure timeout."""


def is_ros_source(source):
    """True for `ros2` (default topic) and `ros2:<topic>`."""
    return source == PREFIX[:-1] or source.startswith(PREFIX)


def topic_from_source(source):
    """Topic of a `ros2` / `ros2:<topic>` source string (bare `ros2` is DEFAULT_TOPIC), or None for any other source."""
    if not is_ros_source(source):
        return None
    if source == PREFIX[:-1]:
        return DEFAULT_TOPIC
    topic = source[len(PREFIX):]
    if not topic or topic.isspace():
        raise ValueError(f"ROS 2 source needs a topic after 'ros2:' (for example ros2:{DEFAULT_TOPIC}), or use plain 'ros2' for {DEFAULT_TOPIC}")
    return topic


def image_to_bgr(data, height, width, step, encoding):
    """Row-padded `sensor_msgs/Image` bytes to a fresh contiguous (height, width, 3) BGR uint8 array."""
    if encoding not in ENCODINGS:
        raise ValueError(f"Unsupported image encoding '{encoding}'; supported: {', '.join(sorted(ENCODINGS))}")
    channels, convert = ENCODINGS[encoding]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")
    if step < width * channels:
        raise ValueError(f"Image step {step} is smaller than one {encoding} row ({width * channels} bytes)")
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError(f"Image data has {raw.size} bytes, expected {height * step} for {width}x{height} {encoding}")
    rows = raw[:height * step].reshape(height, step)[:, :width * channels].reshape(height, width, channels)
    return np.array(convert(rows), order="C")


class LatestFrame:
    """Thread-safe single slot: a new put replaces any frame the consumer has not taken yet."""

    def __init__(self, warn_after=2.0, timeout=10.0, describe=lambda: ""):
        if not 0 < warn_after < timeout:
            raise ValueError("Stale-frame warning time must be positive and below the failure timeout")
        self.warn_after, self.timeout, self.describe = warn_after, timeout, describe
        self._condition = threading.Condition()
        self._item = None
        self._closed = False
        self.received = 0
        self.dropped = 0

    def put(self, item):
        with self._condition:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self.received += 1
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def get(self):
        """Block for the next unseen item. Warns once when stale, raises SourceTimeout when it stays stale."""
        waiting_since = time.monotonic()
        warned = False
        with self._condition:
            while self._item is None:
                if self._closed:
                    raise RuntimeError("ROS 2 source was closed")
                waited = time.monotonic() - waiting_since
                if waited >= self.timeout:
                    raise SourceTimeout(f"No frame for {waited:.1f} s{self.describe()}")
                if waited >= self.warn_after and not warned:
                    log.warning("No frame for %.1f s%s; failing after %.1f s", waited, self.describe(), self.timeout)
                    warned = True
                limit = self.timeout if warned else self.warn_after
                self._condition.wait(min(0.25, max(limit - waited, 0.01)))
            item, self._item = self._item, None
            return item


class RosImageSource:
    """Subscribes to an Image topic on a private node spun by a background thread.

    `read()` returns `(bgr_frame, stamp_seconds)` for the newest frame not yet returned; older frames are dropped so a
    slow consumer never accumulates latency. `stamp_seconds` is the message header stamp.
    """

    def __init__(self, topic, reliable=False, warn_after=2.0, timeout=10.0):
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
            from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
            from rclpy.signals import SignalHandlerOptions
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError(
                f"ROS 2 Python (rclpy, sensor_msgs) is not importable ({exc}). Run `source /opt/ros/jazzy/setup.bash` "
                "and use a Python that can see it (system python3.12, or a venv created with --system-site-packages)") from exc
        self.topic = topic
        self._shutdown_error = ExternalShutdownException
        self._frames = LatestFrame(warn_after, timeout, describe=self._describe)
        self._context = Context()
        # Python keeps its own SIGINT/SIGTERM handling so the caller can shut down through KeyboardInterrupt.
        rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)
        try:
            self._node = rclpy.create_node(f"orion_image_source_{id(self) & 0xffff:04x}", context=self._context)
            qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE if reliable else QoSReliabilityPolicy.BEST_EFFORT,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=1)
            self._subscription = self._node.create_subscription(Image, topic, self._frames.put, qos)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
        except Exception:
            self._context.try_shutdown()
            raise
        self._thread = threading.Thread(target=self._spin, name="orion-ros-spin", daemon=True)
        self._thread.start()
        self._closed = False
        self.encoding = None  # encoding of the most recent frame returned by read()
        log.info("Subscribed to %s (%s, keep last 1)", topic, "reliable" if reliable else "best effort")

    def _spin(self):
        try:
            self._executor.spin()
        except self._shutdown_error:
            pass  # the context was shut down by release()

    def _describe(self):
        try:
            publishers = self._node.count_publishers(self.topic)
        except Exception:
            return f" on {self.topic}"
        return f" on {self.topic} ({publishers} publisher{'s' if publishers != 1 else ''} visible)"

    def read(self):
        message = self._frames.get()
        frame = image_to_bgr(message.data, message.height, message.width, message.step, message.encoding)
        self.encoding = message.encoding
        stamp = message.header.stamp
        return frame, stamp.sec + stamp.nanosec * 1e-9

    def release(self):
        if self._closed:
            return
        self._closed = True
        self._frames.close()
        self._context.try_shutdown()
        self._thread.join(timeout=5)
        self._node.destroy_node()
        log.info("Stopped ROS 2 source %s: %d frames received, %d dropped", self.topic, self._frames.received, self._frames.dropped)
