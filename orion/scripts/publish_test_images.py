#!/usr/bin/env python3
"""Publish a video file or a synthetic moving pattern as sensor_msgs/Image, standing in for a RoverFlake2 camera.

By default it imitates the output of main's cameras_cpp decoder (`/{camera_name}/image_raw_decoded`): 640x480 `rgb8`
(the v4l2_camera default, which the ffmpeg decoder passes through) at 30 Hz with header.stamp = now. Orion's
`--source ros2` can then be tested without a camera:

    source /opt/ros/jazzy/setup.bash
    python3 scripts/publish_test_images.py [--topic /realsense435i/image_raw_decoded] [--video clip.mp4] [--frames 300]

`--encoding yuv422_yuy2` imitates a raw v4l2_camera topic configured for YUYV output, `--encoding bgr8` an older
decoder. `--topic /camera/color/image_raw` imitates the unmerged cameras.cpp RealSense publisher. Needs rclpy and
numpy; --video also needs OpenCV (cv2).
"""
import argparse
import sys

import numpy as np


WIDTH, HEIGHT = 640, 480
ENCODINGS = ("rgb8", "bgr8", "yuv422_yuy2")


def rgb_to_yuyv(rgb):
    """(h, w, 3) RGB uint8 to packed Y U Y V rows (BT.601 limited range, chroma averaged over pixel pairs)."""
    r, g, b = (rgb[:, :, i].astype(np.float32) for i in range(3))
    y = 16 + 0.257 * r + 0.504 * g + 0.098 * b
    u = 128 - 0.148 * r - 0.291 * g + 0.439 * b
    v = 128 + 0.439 * r - 0.368 * g - 0.071 * b
    u = (u[:, 0::2] + u[:, 1::2]) / 2
    v = (v[:, 0::2] + v[:, 1::2]) / 2
    packed = np.empty((rgb.shape[0], rgb.shape[1] // 2, 4), dtype=np.uint8)
    packed[:, :, 0] = np.clip(y[:, 0::2] + 0.5, 0, 255)
    packed[:, :, 1] = np.clip(u + 0.5, 0, 255)
    packed[:, :, 2] = np.clip(y[:, 1::2] + 0.5, 0, 255)
    packed[:, :, 3] = np.clip(v + 0.5, 0, 255)
    return packed.reshape(rgb.shape[0], rgb.shape[1] * 2)


def encode(rgb, encoding):
    """RGB frame to (message data bytes, bytes per row) for the requested ROS encoding."""
    if encoding == "bgr8":
        return np.ascontiguousarray(rgb[:, :, ::-1]).tobytes(), WIDTH * 3
    if encoding == "yuv422_yuy2":
        return rgb_to_yuyv(rgb).tobytes(), WIDTH * 2
    return rgb.tobytes(), WIDTH * 3


def synthetic_frame(index):
    """Gradient background with a bright block moving across the frame, as (480, 640, 3) RGB uint8."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = (x * 255 // WIDTH + index * 3) % 256
    frame[:, :, 1] = y * 255 // HEIGHT
    frame[:, :, 2] = 96
    left = (index * 8) % (WIDTH - 80)
    top = HEIGHT // 2 - 60 + int(60 * np.sin(index / 15))
    frame[top:top + 120, left:left + 80] = (255, 255, 255)
    return frame


class VideoFrames:
    """Endless RGB frames from a video file, resized to 640x480, looping at the end."""

    def __init__(self, path):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

    def __call__(self, index):
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError("Video has no readable frames")
        frame = self.cv2.resize(frame, (WIDTH, HEIGHT), interpolation=self.cv2.INTER_AREA)
        return np.ascontiguousarray(frame[:, :, ::-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", default="/realsense435i/image_raw_decoded")
    parser.add_argument("--encoding", choices=ENCODINGS, default="rgb8", help="Encoding of the published images")
    parser.add_argument("--video", help="Video file to loop; default is a synthetic moving pattern")
    parser.add_argument("--rate", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--frames", type=int, help="Stop after this many frames (default: run until Ctrl-C)")
    parser.add_argument("--frame-id", default="camera")
    args = parser.parse_args()
    if args.rate <= 0 or (args.frames is not None and args.frames <= 0):
        parser.error("--rate and --frames must be positive")
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from sensor_msgs.msg import Image
    except ImportError as exc:
        print(f"rclpy is not importable ({exc}); run `source /opt/ros/jazzy/setup.bash` first", file=sys.stderr)
        return 2
    source = VideoFrames(args.video) if args.video else synthetic_frame

    rclpy.init()
    node = rclpy.create_node("orion_test_image_publisher")
    publisher = node.create_publisher(Image, args.topic, 10)
    sent = 0

    def publish():
        nonlocal sent
        frame = source(sent)
        message = Image()
        message.header.stamp = node.get_clock().now().to_msg()
        message.header.frame_id = args.frame_id
        message.height, message.width = HEIGHT, WIDTH
        message.encoding = args.encoding
        message.is_bigendian = 0
        message.data, message.step = encode(frame, args.encoding)
        publisher.publish(message)
        sent += 1
        if args.frames is not None and sent >= args.frames:
            rclpy.shutdown()

    node.create_timer(1.0 / args.rate, publish)
    node.get_logger().info(f"Publishing {WIDTH}x{HEIGHT} {args.encoding} at {args.rate:g} Hz on {args.topic}")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f"Published {sent} frames on {args.topic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
