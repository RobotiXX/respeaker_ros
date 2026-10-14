"""Check PCM fidelity and planar direction quaternions without USB hardware."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import rclpy.time
import tf_transformations as transforms
from respeaker_ros.respeaker_node import RespeakerNode


class SensorOutputTest(unittest.TestCase):
    def test_signed_pcm_round_trip(self):
        publisher = Mock()
        publisher.get_subscription_count.return_value = 1
        node = SimpleNamespace(
            _pub_audio_channels={0: publisher}, _pub_audio=publisher,
            main_channel=SimpleNamespace(value=0), is_speaking=False,
            speech_prefetch_buffer=np.zeros(32, dtype=np.uint8))
        samples = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
        RespeakerNode.on_audio(node, samples, 0)
        msg = publisher.publish.call_args.args[0]
        decoded = np.frombuffer(bytes(msg.data), dtype='<i2')
        np.testing.assert_array_equal(decoded, samples)
        self.assertEqual(len(msg.data), 2 * len(samples))

    def test_direction_is_yaw_with_ros_xyzw_order(self):
        publisher = Mock()
        now = rclpy.time.Time(seconds=2)
        node = SimpleNamespace(
            get_clock=lambda: SimpleNamespace(now=lambda: now),
            respeaker=SimpleNamespace(is_voice=lambda: 0, direction=180),
            doa_yaw_offset=SimpleNamespace(value=90),
            doa_xy_offset=SimpleNamespace(value=0),
            sensor_frame_id=SimpleNamespace(value='respeaker_base'),
            prev_is_voice=0, prev_doa=None, _pub_doa_raw=Mock(),
            _pub_doa=publisher, is_speaking=False,
            speech_stopped=rclpy.time.Time(seconds=0),
            speech_continuation=SimpleNamespace(value=0.5))
        RespeakerNode.on_timer(node)
        q = publisher.publish.call_args.args[0].pose.orientation
        roll, pitch, yaw = transforms.euler_from_quaternion([q.x, q.y, q.z, q.w])
        np.testing.assert_allclose([roll, pitch, yaw], [0, 0, np.pi / 2], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
