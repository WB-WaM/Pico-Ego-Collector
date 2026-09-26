"""Missing raw hands must warn, retain episodes, and export invalid zero sentinels."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_lerobot_v3 as export
import serve_annotator as server


POSE = "1,2,3,0,0,0,1"
HAND = {"isActive": 1, "HandJointLocations": [{"p": POSE}] * 26}


class MissingHandTests(unittest.TestCase):
    def load(self, hands):
        records = [{"notice": "test", "timeStampNs": 1_000_000_000}]
        for i, hand in enumerate(hands):
            record = {"timeStampNs": 1_000_000_000 + i * 100_000_000,
                      "Head": {"pose": POSE}, "Body": {"joints": [{"p": POSE}] * 24}}
            if hand is not None:
                record["Hand"] = hand
            records.append(record)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tracking.txt"
            path.write_text("\n".join(json.dumps(r) for r in records))
            return export.load_tracking_arrays(path)

    def check(self, tracking):
        episodes = [{"start": 0., "end": .3, "prompt": "test task"}]
        kept, invalid = export.filter_episodes_by_joint_validity(episodes, tracking, 10)
        self.assertEqual(kept, episodes)
        return export.hand_export_warning(invalid, 3)

    def test_no_hand_field_keeps_all_episodes_and_zeros(self):
        tracking = self.load([None] * 3)
        warning = self.check(tracking)
        self.assertEqual(warning["empty_frames"], [3, 3])
        self.assertEqual(warning["episodes_affected"], 1)
        g1 = {"times": tracking["times"], "qpos": np.zeros((3, 36), np.float32)}
        for t in (0., .1, .2):
            observation = export.sample_observation(t, tracking, g1)
            np.testing.assert_array_equal(observation["observation.pico_hands26_pose"], 0)
            np.testing.assert_array_equal(observation["observation.pico_hand_active"], 0)
            np.testing.assert_array_equal(observation["observation.pico_joint_valid"][25:], 0)
            np.testing.assert_array_equal(observation["observation.pico_joint_valid"][:25], 1)

    def test_missing_left_preserves_observed_right(self):
        tracking = self.load([{"rightHand": HAND}] * 3)
        warning = self.check(tracking)
        self.assertEqual(warning["empty_frames"], [3, 0])
        poses = tracking["hands26_pose"].reshape(3, 2, 26, 7)
        np.testing.assert_array_equal(poses[:, 0], 0)
        np.testing.assert_array_equal(poses[:, 1, 0], [[1, 2, 3, 0, 0, 0, 1]] * 3)
        np.testing.assert_array_equal(tracking["joint_valid"][:, 51:], 1)

    def test_complete_and_inactive_finite_hands_unchanged(self):
        inactive = copy.deepcopy(HAND)
        inactive["isActive"] = 0
        tracking = self.load([{"leftHand": inactive, "rightHand": HAND}] * 3)
        self.assertIsNone(self.check(tracking))
        np.testing.assert_array_equal(tracking["hand_active"], [[0, 1]] * 3)
        np.testing.assert_array_equal(tracking["joint_valid"], 1)

    def test_short_gap_holds_but_never_becomes_observed(self):
        both = {"leftHand": HAND, "rightHand": HAND}
        tracking = self.load([both, None, both])
        self.assertEqual(self.check(tracking)["empty_frames"], [1, 1])
        np.testing.assert_array_equal(tracking["hands26_pose"][1], tracking["hands26_pose"][0])
        np.testing.assert_array_equal(tracking["joint_valid"][1, 25:], 0)
        np.testing.assert_array_equal(tracking["hand_active"][1], 0)

    def test_quality_counts_exact_selected_timeline(self):
        both = {"leftHand": HAND, "rightHand": HAND}
        tracking = self.load([None, both, both])
        episode = {"start": .1, "end": .3, "prompt": "only complete frames"}
        kept, invalid = export.filter_episodes_by_joint_validity([episode], tracking, 10)
        self.assertEqual(kept, [episode])
        self.assertEqual(invalid, [])

    def test_web_only_passes_explicit_confirmation_and_check_does_not_mark_processed(self):
        for dry_run, allow in [(True, False), (False, False), (False, True), (False, "false")]:
            with self.subTest(dry_run=dry_run, allow=allow):
                handler = server.AnnotatorHandler.__new__(server.AnnotatorHandler)
                handler.path = "/api/export_lerobot"
                handler.state = MagicMock()
                handler.state.root = Path("/tmp/test-pico")
                handler.state.output_dir = Path("/tmp/test-pico/lerobot_v3")
                handler.state.resolve_project_path.side_effect = lambda p: p
                handler.state.session_paths.return_value = {
                    "session_dir": Path("/tmp/test-pico/raw/20260914_162219"),
                    "g1": Path("/tmp/test-pico/g1/a.pkl"),
                }
                handler.state.export_command.side_effect = lambda cmd: cmd
                handler.send_json = MagicMock()
                payload = {"session_id": "20260914_162219", "dry_run": dry_run,
                           "allow_missing_hands": allow}
                with patch.object(server, "read_json_body", return_value=payload), \
                     patch.object(server, "start_job", return_value={}) as start:
                    handler._do_POST()
                args, kwargs = start.call_args
                self.assertEqual("--allow-missing-hands" in args[2], allow is True)
                self.assertEqual(kwargs["on_success"] is None, dry_run)


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "Run integration in the lerobot environment")
class ExportIntegrationTests(unittest.TestCase):
    def test_warning_confirmation_and_official_dataset_readback(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        with tempfile.TemporaryDirectory(prefix="pico_missing_hands_test_") as temp:
            root = Path(temp)
            sid = "20260914_000000"
            raw = root / "raw" / sid
            raw.mkdir(parents=True)
            t0 = 1_000_000_000
            records = [{"notice": "synthetic regression fixture", "timeStampNs": t0}]
            for i in range(60):
                records.append({"timeStampNs": t0 + round(i * 1e9 / 60),
                                "Head": {"pose": POSE}, "Body": {"joints": [{"p": POSE}] * 24}})
            tracking = raw / f"trackingData_{sid}.txt"
            original = "\n".join(json.dumps(r) for r in records)
            tracking.write_text(original)
            writer = cv2.VideoWriter(str(raw / f"CameraRecord_{sid}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 60, (192, 96))
            self.assertTrue(writer.isOpened())
            for i in range(60):
                writer.write(np.full((96, 192, 3), i * 3, dtype=np.uint8))
            writer.release()
            with (root / f"{sid}_unitree_g1.pkl").open("wb") as f:
                pickle.dump({"root_pos": np.zeros((60, 3)), "root_rot": np.tile([0, 0, 0, 1], (60, 1)),
                             "dof_pos": np.zeros((60, 29)), "times": np.arange(60) / 60,
                             "time_source": "camera_first_frame_timestamp_ns", "time_origin_ns": t0}, f)
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps({"session_id": sid, "episodes": [
                {"start": 0., "end": .25, "prompt": "test"},
                {"start": .5, "end": .75, "prompt": "test"}]}))
            with patch.object(sys, "argv", ["export", "--session", sid, "--annotations", str(annotations),
                                            "--raw-dir", str(root / "raw"), "--g1-dir", str(root),
                                            "--output-dir", str(root / "output"), "--repo-id", "test/missing",
                                            "--preprocess-workers", "1", "--image-writer-threads", "1",
                                            "--encoder-threads", "1", "--dry-run"]):
                args = export.parse_args()
            summary = export.export_dataset(args)
            self.assertEqual(summary["episodes"], 2)
            self.assertEqual(summary["frames"], 30)
            self.assertTrue(summary["requires_hand_confirmation"])
            self.assertFalse(summary["episode_out_of_range"])
            self.assertFalse(args.output_dir.exists())
            args.dry_run = False
            with self.assertRaisesRegex(SystemExit, "allow-missing-hands"):
                export.export_dataset(args)
            self.assertFalse(args.output_dir.exists())
            args.allow_missing_hands = True
            summary = export.export_dataset(args)
            self.assertEqual(summary["written_frames"], 30)
            dataset = LeRobotDataset(repo_id="test/missing", root=args.output_dir, video_backend="pyav")
            self.assertEqual(len(dataset), 30)
            self.assertEqual(dataset.num_episodes, 2)
            for row in dataset.hf_dataset:
                np.testing.assert_array_equal(row["observation.pico_hands26_pose"], 0)
                np.testing.assert_array_equal(row["observation.pico_hand_active"], 0)
                np.testing.assert_array_equal(row["observation.pico_joint_valid"][25:], 0)
                np.testing.assert_array_equal(row["observation.pico_joint_valid"][:25], 1)
            for i in (0, 14, 15, 29):
                self.assertEqual(tuple(dataset[i][export.IMAGE_KEY_STEREO_LEFT].shape), (3, 96, 96))
            meta = json.loads((args.output_dir / "meta/pico_ego_release.json").read_text())
            self.assertTrue(meta["episode_quality_policy"]["missing_hands_confirmed"])
            self.assertEqual(meta["quality_control"]["episodes_kept"], 2)
            self.assertEqual(tracking.read_text(), original)


if __name__ == "__main__":
    unittest.main()
