# Pipeline reference

See the [README](../README.md) for installation and web usage, and the
[data format](lerobot_v3_release_spec.md) for exported fields.
Run commands from the repository root, replacing `SESSION` with a recording ID.

## Raw data

Each recording needs a pair of files:

```text
raw/SESSION/
  trackingData_SESSION.txt
  CameraRecord_SESSION.mp4
```

Tracking uses JSON Lines. The first line contains the camera's first-frame `timeStampNs`,
`cameraIntrinsics`, and `cameraExtrinsics`. Subsequent lines contain `timeStampNs`,
`Head.pose`, `Body.joints[24]`, and `Hand.leftHand` / `Hand.rightHand`
(`isActive`, `HandJointLocations[26]`). Each pose is a comma-separated
`x,y,z,qx,qy,qz,qw` string; body and hand joints store it in `p`.
Video contains side-by-side stereo images. The head pose describes the headset, not the camera.

Body joint order:

```text
Pelvis, Left_Hip, Right_Hip, Spine1, Left_Knee, Right_Knee,
Spine2, Left_Ankle, Right_Ankle, Spine3, Left_Foot, Right_Foot,
Neck, Left_Collar, Right_Collar, Head, Left_Shoulder, Right_Shoulder,
Left_Elbow, Right_Elbow, Left_Wrist, Right_Wrist, Left_Hand, Right_Hand
```

Pico hands follow OpenXR ordering: `Palm=0`, `Wrist=1`. Conversion to MediaPipe's 21 points
removes the palm and the four non-thumb metacarpals, retaining the wrist and finger chains.

## Import

```bash
conda activate pico
python scripts/sync_pico_download.py --keep-source
# A local Download copy or an explicit MTP path
python scripts/sync_pico_download.py --source /path/to/Download --date 20260917 --keep-source
```

`--keep-source` preserves device originals. Without it, paired files are deleted from the
device after successful copying. Unpaired files are copied but kept on the device.

Any device is accepted by default. For stable prefixes across multiple devices, create
`configs/pico_devices.local.json` (ignored by Git). Keys are the identifiers after
`host=` in the MTP path:

```json
{"MY_PICO_DEVICE_ID": "group0", "ANOTHER_PICO_DEVICE_ID": "group1"}
```

`group0` has no prefix; other groups prepend `groupN_` to filenames.
When this configuration exists, unregistered devices are rejected.

## G1 and Wuji retargeting

```bash
conda activate pico
python scripts/retarget_pico_to_g1.py --session SESSION
python scripts/retarget_pico_hand_to_wuji.py --session SESSION --hand both \
  --config configs/wuji_retarget_pico_customer.yaml
python scripts/render_g1_mujoco_preview.py --session SESSION
python scripts/visualize_wuji_qpos_mujoco.py --session SESSION --hand both --render-mode mjcf
```

G1 uses GMR's default `xrobot → unitree_g1` solver. The web server's optional Wuji step uses
`wuji_retarget_pico_best.yaml`; the standalone command can select the `customer` configuration.
Wuji trajectories are saved as separate pkl files and are not consumed by the base LeRobot exporter.

G1 pkl files contain `root_pos (T,3)`, `root_rot (T,4), xyzw`, `dof_pos (T,29)`,
`times`, `time_origin_ns`, and `time_source=camera_first_frame_timestamp_ns`.
Regenerate older files that lack the timing fields.

## CLI export

Save episode annotations in the web interface first, then run:

```bash
conda activate lerobot
python scripts/export_lerobot_v3.py \
  --session SESSION \
  --annotations /tmp/pico_ego_collector/SESSION/annotations/SESSION_episodes.json \
  --output-dir lerobot_v3/SESSION \
  --repo-id pico_ego/SESSION --dry-run
```

Remove `--dry-run` to export after validation. Add `--allow-missing-hands` to accept missing
hand observations, or `--overwrite` to replace existing output. Adjust the annotation path
if you set `PICO_EGO_TMP`. Run with `--help` for all options.
The legacy `consolidate_lerobot_v3.py` accepts schema v1 only; it is not part of the current v2 export workflow.

## Working files and checks

The default work directory is `/tmp/pico_ego_collector/`; use `PICO_EGO_TMP` for persistent
storage. Each session contains `annotations/`, `retargeted/`, `wuji_retargeted/`, and preview
caches. Export keeps annotations. **Redo** clears generated files and preserves raw data;
**Delete data** also removes the corresponding raw recording.

Core regression tests use temporary fixtures and need no connected device:

```bash
conda run -n pico python -m unittest discover -s tests -p 'test_*.py'
conda run -n pico python scripts/test_pico_wuji_mapping.py
conda run -n lerobot python -m unittest discover -s tests -p test_export_missing_hands.py
# Optional web logic tests, if Node.js is installed
node --test tests/test_annotator_import.cjs tests/test_annotator_export.cjs
```
