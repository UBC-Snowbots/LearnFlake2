import logging
import threading
import time

import numpy as np
import pytest

from orion_bench import ros_source
from orion_bench.ros_source import (DEFAULT_TOPIC, LatestFrame, SourceTimeout, image_to_bgr, is_ros_source,
                                    topic_from_source)


def rgb_pattern(height=4, width=6):
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def test_rgb8_becomes_bgr():
    rgb = rgb_pattern()
    bgr = image_to_bgr(rgb.tobytes(), 4, 6, 18, "rgb8")
    assert bgr.shape == (4, 6, 3) and bgr.dtype == np.uint8 and bgr.flags["C_CONTIGUOUS"]
    assert np.array_equal(bgr, rgb[:, :, ::-1])


def test_bgr8_is_unchanged_and_writable():
    bgr = rgb_pattern()
    out = image_to_bgr(bgr.tobytes(), 4, 6, 18, "bgr8")
    assert np.array_equal(out, bgr)
    out[0, 0, 0] = 255  # frames get overlays drawn on them, even when the message buffer is read-only


def test_mono8_is_replicated_to_three_channels():
    grey = np.arange(24, dtype=np.uint8).reshape(4, 6)
    bgr = image_to_bgr(grey.tobytes(), 4, 6, 6, "mono8")
    assert bgr.shape == (4, 6, 3) and bgr.flags["C_CONTIGUOUS"] and bgr.flags["WRITEABLE"]
    for channel in range(3):
        assert np.array_equal(bgr[:, :, channel], grey)


def test_rgba8_and_bgra8_drop_alpha():
    rgba = np.concatenate([rgb_pattern(), np.full((4, 6, 1), 200, np.uint8)], axis=2)
    assert np.array_equal(image_to_bgr(rgba.tobytes(), 4, 6, 24, "rgba8"), rgba[:, :, 2::-1])
    assert np.array_equal(image_to_bgr(rgba.tobytes(), 4, 6, 24, "bgra8"), rgba[:, :, :3])


def test_row_padding_in_step_is_skipped():
    rgb = rgb_pattern()
    padded = np.zeros((4, 24), dtype=np.uint8)
    padded[:, :18] = rgb.reshape(4, 18)
    padded[:, 18:] = 99
    assert np.array_equal(image_to_bgr(bytearray(padded.tobytes()), 4, 6, 24, "rgb8"), rgb[:, :, ::-1])
    grey = np.arange(24, dtype=np.uint8).reshape(4, 6)
    padded = np.full((4, 8), 99, dtype=np.uint8)
    padded[:, :6] = grey
    assert np.array_equal(image_to_bgr(padded.tobytes(), 4, 6, 8, "mono8")[:, :, 1], grey)


def yuyv_row(*pixels):
    """Pairs of (y, u, v) triples to Y U Y V bytes, taking the chroma of the first pixel of each pair."""
    out = []
    for first, second in zip(pixels[0::2], pixels[1::2]):
        out += [first[0], first[1], second[0], first[2]]
    return out


def test_yuyv_known_colours():
    # BT.601 limited range: black, white, grey and a saturated red; chroma shared by each pixel pair
    black, white, grey, red = (16, 128, 128), (235, 128, 128), (128, 128, 128), (81, 90, 240)
    data = np.array(yuyv_row(black, black, white, white, grey, grey, red, red), dtype=np.uint8)
    bgr = image_to_bgr(data.tobytes(), 1, 8, 16, "yuv422_yuy2")
    assert bgr.shape == (1, 8, 3) and bgr.dtype == np.uint8
    assert bgr[0, 0].tolist() == [0, 0, 0] and bgr[0, 2].tolist() == [255, 255, 255]
    assert np.abs(bgr[0, 4].astype(int) - 130).max() <= 1
    assert np.abs(bgr[0, 6].astype(int) - [0, 0, 254]).max() <= 2
    assert np.array_equal(image_to_bgr(data.tobytes(), 1, 8, 16, "yuyv"), bgr)


def test_uyvy_is_the_byte_swapped_yuyv():
    rng = np.random.default_rng(1)
    yuyv = rng.integers(0, 256, size=(3, 8), dtype=np.uint8)
    uyvy = yuyv.reshape(3, 4, 2)[:, :, ::-1].reshape(3, 8)  # swap bytes within each 16-bit pixel
    assert np.array_equal(image_to_bgr(uyvy.tobytes(), 3, 4, 8, "yuv422"), image_to_bgr(yuyv.tobytes(), 3, 4, 8, "yuv422_yuy2"))


def test_yuyv_matches_opencv_and_handles_padding():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    yuyv = rng.integers(0, 256, size=(6, 16), dtype=np.uint8)  # 6 rows, 8 pixels
    padded = np.full((6, 20), 7, dtype=np.uint8)
    padded[:, :16] = yuyv
    ours = image_to_bgr(padded.tobytes(), 6, 8, 20, "yuv422_yuy2")
    reference = cv2.cvtColor(yuyv.reshape(6, 8, 2), cv2.COLOR_YUV2BGR_YUY2)
    assert np.abs(ours.astype(int) - reference.astype(int)).max() <= 3


def test_odd_width_yuyv_is_rejected():
    with pytest.raises(ValueError, match="even"):
        image_to_bgr(bytes(4 * 3 * 2), 4, 3, 6, "yuv422_yuy2")


@pytest.mark.parametrize("encoding", ["16UC1", "32FC1", "bayer_rggb8", "nv12", ""])
def test_unsupported_encoding_names_the_encoding(encoding):
    with pytest.raises(ValueError, match=f"'{encoding}'"):
        image_to_bgr(bytes(100), 4, 6, 12, encoding)


def test_inconsistent_geometry_is_rejected():
    with pytest.raises(ValueError, match="step"):
        image_to_bgr(bytes(100), 4, 6, 17, "rgb8")
    with pytest.raises(ValueError, match="bytes"):
        image_to_bgr(bytes(4 * 18 - 1), 4, 6, 18, "rgb8")
    with pytest.raises(ValueError, match="size"):
        image_to_bgr(bytes(0), 0, 6, 18, "rgb8")


def test_topic_parsing():
    assert DEFAULT_TOPIC == "/realsense435i/image_raw_decoded"
    assert topic_from_source("ros2") == DEFAULT_TOPIC
    assert topic_from_source("ros2:/camera/color/image_raw") == "/camera/color/image_raw"
    assert topic_from_source("ros2:/cam_0") == "/cam_0"
    assert topic_from_source("ros2:/front_cam/image_raw_decoded") == "/front_cam/image_raw_decoded"
    for other in ("0", "/dev/video0", "rtsp://host/stream", "clip.mp4", "ros2.mp4"):
        assert topic_from_source(other) is None and not is_ros_source(other)
    with pytest.raises(ValueError, match="topic"):
        topic_from_source("ros2:")


def test_live_demo_treats_ros2_as_live_source():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from orion_bench.live_demo import live_source
    assert live_source("ros2") and live_source("ros2:/camera/color/image_raw")
    assert not live_source("clip.mp4")


def test_latest_frame_keeps_only_the_newest():
    frames = LatestFrame(warn_after=1, timeout=2)
    for index in range(5):
        frames.put(index)
    assert frames.get() == 4
    assert frames.received == 5 and frames.dropped == 4


def test_latest_frame_blocks_until_a_new_frame():
    frames = LatestFrame(warn_after=1, timeout=5)
    threading.Timer(0.1, frames.put, args=("late",)).start()
    start = time.monotonic()
    assert frames.get() == "late"
    assert 0.05 < time.monotonic() - start < 2


class Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_latest_frame_warns_then_times_out():
    frames = LatestFrame(warn_after=0.1, timeout=0.4, describe=lambda: " on /nothing (0 publishers visible)")
    handler, logger = Collect(), logging.getLogger("orion.ros_source")
    logger.addHandler(handler)  # attached directly: some ROS pytest plugins reconfigure root logging
    try:
        with pytest.raises(SourceTimeout, match="/nothing"):
            frames.get()
    finally:
        logger.removeHandler(handler)
    assert len(handler.messages) == 1 and "No frame" in handler.messages[0]


def test_latest_frame_rejects_bad_limits():
    with pytest.raises(ValueError):
        LatestFrame(warn_after=5, timeout=1)


def test_closed_source_wakes_a_waiting_reader():
    frames = LatestFrame(warn_after=1, timeout=5)
    threading.Timer(0.1, frames.close).start()
    with pytest.raises(RuntimeError, match="closed"):
        frames.get()


def test_rclpy_round_trip():
    pytest.importorskip("rclpy")
    pytest.importorskip("sensor_msgs")
    import rclpy
    from rclpy.context import Context
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import Image

    topic = f"/orion_test_{time.monotonic_ns()}"
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node("orion_test_publisher", context=context)
    publisher = node.create_publisher(Image, topic, QoSProfile(depth=10))
    source = ros_source.RosImageSource(topic, warn_after=2, timeout=10)
    rgb = rgb_pattern(8, 10)
    stop = threading.Event()

    def publish():
        count = 0
        while not stop.is_set():
            message = Image()
            message.header.stamp.sec, message.header.stamp.nanosec = 100 + count, 500_000_000
            message.height, message.width, message.step, message.encoding = 8, 10, 30, "rgb8"
            message.data = rgb.tobytes()
            publisher.publish(message)
            count += 1
            time.sleep(0.05)

    thread = threading.Thread(target=publish)
    thread.start()
    try:
        frame, stamp = source.read()
        assert source.encoding == "rgb8"
        assert np.array_equal(frame, rgb[:, :, ::-1])
        assert stamp >= 100.5 and stamp % 1 == pytest.approx(0.5)
    finally:
        stop.set()
        thread.join()
        source.release()
        node.destroy_node()
        context.try_shutdown()
