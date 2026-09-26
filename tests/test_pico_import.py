"""Regress stale MTP mounts and stale import manifests without device writes."""
import errno
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sync_pico_download as sync
import serve_annotator as server


class ImportTests(unittest.TestCase):
    def test_localized_storage_path_remains_discoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = "\u5185\u90e8\u5171\u4eab\u5b58\u50a8\u7a7a\u95f4"
            download = root / "1000/gvfs/mtp:host=TEST_PICO" / storage / "Download"
            download.mkdir(parents=True)
            self.assertEqual(sync.find_pico_download(root, ()), download)

    def test_legacy_csv_status_is_preserved_by_server_and_importer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = []
            for sid in ("20260917_120000", "20260917_130000"):
                session = root / "raw" / sid
                session.mkdir(parents=True)
                video = session / f"CameraRecord_{sid}.mp4"
                video.write_bytes(b"fixture")
                names.append(video.name)
            path = root / "pico_processing_status.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sync.STATUS_FIELDS)
                writer.writeheader()
                for name, status in zip(names, ("\u5df2\u5904\u7406", "\u672a\u5904\u7406")):
                    writer.writerow({"file_name": name, "device_group": "\u672a\u5206\u7ec4", "status": status})
            original = path.read_bytes()
            state = server.AnnotatorState(root, tmp_root=root / "work")
            rows = state.load_processing_status()
            self.assertEqual(rows[names[0]]["status"], "processed")
            self.assertEqual(rows[names[1]]["status"], "unprocessed")
            self.assertEqual(rows[names[0]]["device_group"], "ungrouped")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(state.ensure_processing_status("20260917_120000"), "processed")
            # Reimporting must not reset a legacy completed recording.
            sync.update_processing_status([{
                "paired": True, "video_file": names[0], "tracking_file": "trackingData_20260917_120000.txt",
                "device_group": "group0",
            }], path)
            with path.open(encoding="utf-8") as f:
                written = {row["file_name"]: row for row in csv.DictReader(f)}
            self.assertEqual(written[names[0]]["status"], "processed")
            self.assertEqual(written[names[1]]["status"], "unprocessed")
            self.assertEqual(written[names[1]]["device_group"], "ungrouped")

    def test_fresh_checkout_imports_unregistered_device_and_keeps_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mtp:host=NEW_PICO/Download"
            source.mkdir(parents=True)
            sid = "20260917_123456"
            tracking = source / f"trackingData_{sid}.txt"
            video = source / f"CameraRecord_{sid}.mp4"
            tracking.write_text("tracking fixture")
            video.write_bytes(b"video fixture")
            with patch.object(sync, "DEVICE_GROUPS_FILE", root / "absent.json"):
                rows = sync.sync_pairs(source, root / "raw", delete_source=False)
            self.assertEqual(rows[0]["device_group"], "group0")
            self.assertEqual((root / "raw" / sid / tracking.name).read_bytes(), tracking.read_bytes())
            self.assertEqual((root / "raw" / sid / video.name).read_bytes(), video.read_bytes())

    def test_local_device_mapping_preserves_group_and_rejects_unknown_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = Path(tmp) / "devices.json"
            mapping.write_text(json.dumps({"MY_PICO": "group2"}))
            with patch.object(sync, "DEVICE_GROUPS_FILE", mapping):
                self.assertEqual(sync.infer_device_group(Path("mtp:host=MY_PICO/Download")), "group2")
                with self.assertRaisesRegex(ValueError, "Unknown Pico device"):
                    sync.infer_device_group(Path("mtp:host=OTHER_PICO/Download"))

    def test_invalid_device_mapping_fails_before_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = root / "devices.json"
            mapping.write_text(json.dumps({"MY_PICO": "../../outside"}))
            with patch.object(sync, "DEVICE_GROUPS_FILE", mapping):
                with self.assertRaisesRegex(ValueError, "Invalid device mapping"):
                    sync.sync_pairs(Path("mtp:host=MY_PICO/Download"), root / "raw")
            self.assertFalse((root / "raw").exists())

    def test_conda_discovery_uses_current_installation(self):
        with patch.dict(server.os.environ, {"CONDA_EXE": "/opt/conda/bin/conda"}, clear=True):
            self.assertEqual(server.default_conda_bin(), Path("/opt/conda/bin/conda"))
        with patch.dict(server.os.environ, {"PICO_CONDA_BIN": "/custom/conda", "CONDA_EXE": "/opt/conda/bin/conda"}, clear=True):
            self.assertEqual(server.default_conda_bin(), Path("/custom/conda"))

    def test_bad_mount_does_not_hide_another_connected_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = root / "1000/gvfs/mtp:host=A"
            good = root / "1000/gvfs/mtp:host=B/Internal shared storage/Download"
            broken.mkdir(parents=True)
            good.mkdir(parents=True)
            original_scandir = sync.os.scandir

            def probe(path):
                if str(path).startswith(str(broken)):
                    raise OSError(errno.EIO, "Input/output error", str(path))
                return original_scandir(path)

            with patch.object(sync.os, "scandir", side_effect=probe):
                self.assertEqual(sync.find_pico_download(root, ()), good)

    def test_disconnected_mount_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1000/gvfs/mtp:host=Pico").mkdir(parents=True)
            with patch.object(sync.os, "scandir", side_effect=OSError(errno.EIO, "Input/output error")):
                with self.assertRaisesRegex(OSError, "reconnect USB"):
                    sync.find_pico_download(root, ())

    def test_empty_storage_mount_is_not_a_successful_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1000/gvfs/mtp:host=Pico").mkdir(parents=True)
            with self.assertRaisesRegex(OSError, "internal storage"):
                sync.find_pico_download(root, ())

    def test_failed_import_never_reuses_old_or_partial_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "manifest.csv"
            old.write_text("session_id,paired\n20260914_162219,True\n")
            state = server.AnnotatorState(root, tmp_root=root)

            def fail(cmd, **kwargs):
                manifest = Path(cmd[cmd.index("--manifest") + 1])
                self.assertNotEqual(manifest, old)
                # Even a current incomplete result must not claim success.
                sync.write_manifest([{"session_id": "20260916_123456", "paired": True}], manifest)
                return subprocess.CompletedProcess(cmd, 1, "", "MTP Input/output error")

            with patch.object(server.subprocess, "run", side_effect=fail):
                result = state.run_sync({"date": "20260916"})
            self.assertFalse(result["ok"])
            self.assertEqual(result["rows"], [])
            self.assertEqual(result["imported_session_ids"], [])
            self.assertEqual(result["warning_count"], 0)
            self.assertIn("20260914_162219", old.read_text())

    def test_success_requires_its_own_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = server.AnnotatorState(root, tmp_root=root)
            with patch.object(server.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
                result = state.run_sync({"date": "20260916"})
            self.assertFalse(result["ok"])
            self.assertIn("did not create a manifest", result["stderr"])

    def test_real_import_subprocess_selects_only_requested_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mtp:host=TEST_PICO/Download"
            source.mkdir(parents=True)
            for sid in ("20260914_123456", "20260916_123456"):
                (source / f"trackingData_{sid}.txt").write_text("fixture")
                (source / f"CameraRecord_{sid}.mp4").write_bytes(b"fixture")
            state = server.AnnotatorState(server.PROJECT_ROOT, raw_dir=root / "raw", tmp_root=root / "temp")
            with patch.dict(server.os.environ, {"PICO_DEVICE_GROUPS_FILE": str(root / "no_mapping.json")}):
                first = state.run_sync({"date": "20260916", "source": str(source)})
                second = state.run_sync({"date": "20260916", "source": str(source)})
            self.assertTrue(first["ok"], first["stderr"])
            self.assertEqual(first["imported_session_ids"], ["20260916_123456"])
            self.assertTrue(second["ok"], second["stderr"])
            self.assertEqual(second["rows"], [])
            self.assertNotEqual(first["manifest"], second["manifest"])
            self.assertTrue((source / "trackingData_20260914_123456.txt").exists())
            self.assertFalse((root / "raw/20260914_123456").exists())


if __name__ == "__main__":
    unittest.main()
