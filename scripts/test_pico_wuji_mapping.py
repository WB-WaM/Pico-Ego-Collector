"""Regression checks for the OpenXR wrist/palm boundary in Pico hand retargeting."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from retarget_pico_hand_to_wuji import pico26_to_mediapipe21
from wuji_retargeting.mediapipe import apply_mediapipe_transformations


class PicoWristMappingTest(unittest.TestCase):
    def setUp(self):
        # OpenXR palm lies halfway along the palm, distinct from the wrist.
        self.points = np.zeros((26, 3), dtype=float)
        self.points[0] = [0, .045, 0]  # Palm
        self.points[1] = [0, 0, 0]  # Wrist
        self.points[7] = [-.025, .08, 0]  # Index MCP
        self.points[12] = [0, .09, 0]  # Middle MCP

    def canonical(self, points):
        return apply_mediapipe_transformations(
            pico26_to_mediapipe21(points, "raw"), "right"
        )

    def test_wrist_to_finger_length_is_not_measured_from_palm(self):
        points = self.canonical(self.points)
        self.assertAlmostEqual(np.linalg.norm(points[9] - points[0]), .09)

    def test_palm_noise_does_not_rotate_or_translate_the_retarget_input(self):
        expected = self.canonical(self.points)
        changed = self.points.copy()
        changed[0] += [.02, -.015, .01]
        np.testing.assert_allclose(self.canonical(changed), expected, atol=1e-12)

    def test_world_motion_does_not_change_local_finger_geometry(self):
        rotation = Rotation.from_euler("xyz", [35, -20, 75], degrees=True)
        moved = rotation.apply(self.points) + [1.2, -.4, 2.5]
        np.testing.assert_allclose(self.canonical(moved), self.canonical(self.points), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
