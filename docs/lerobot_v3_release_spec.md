# LeRobot v3 data format

The base exporter uses the official `LeRobotDataset.create()` API with schema
**`pico_ego_v3_schema_v2`**. Defaults are 60 Hz and H.264, with one dataset per session.
Incompatible field, coordinate, or sampling changes require a new schema version.

## Layout and fields

```text
<dataset>/
  data/chunk-*/file-*.parquet
  videos/<camera>/chunk-*/file-*.mp4
  meta/
    info.json
    stats.json
    tasks.parquet
    episodes/chunk-*/file-*.parquet
    pico_ego_release.json
```

| Field | Type / shape | Contents |
|---|---|---|
| `observation.images.stereo_left` | video, `(H,W/2,3)` | Left half of the side-by-side RGB video |
| `observation.images.stereo_right` | video, `(H,W/2,3)` | Right half of the side-by-side RGB video |
| `observation.pico_head_pose` | float32[7] | Headset `xyz + quat_xyzw` |
| `observation.pico_body_pose` | float32[168] | 24 body joints × 7 |
| `observation.pico_hands26_pose` | float32[364] | Left hand, then right hand; 26 points × 7 per hand |
| `observation.g1_qpos` | float32[36] | `[0:3]` root xyz, `[3:7]` root quat xyzw, `[7:36]` G1's 29 joint angles |
| `observation.pico_hand_active` | float32[2] | Left/right tracking status, 0 or 1 |
| `observation.pico_joint_valid` | float32[77] | Per-point validity: head(1), body(24), left hand(26), right hand(26) |

LeRobot also stores `timestamp`, `frame_index`, `episode_index`, global `index`, and
`task_index`. The loader resolves `task` text from the task table and returns images as
normalized CHW tensors. With `--mono`, the full original frame is stored as
`observation.images.pico_ego` instead of splitting it into left/right halves.

This is an observation dataset: **no `action`, aggregated `observation.state`, or Wuji
finger qpos**. Downstream training must define its own state/action representation;
standalone Wuji results are not automatically added to this schema.

## Coordinates

- Pico head, body, and hands retain native **Y-up** world coordinates, with positions in
  meters and quaternions in `xyzw` order.
- G1 qpos uses **Z-up**. Input positions undergo `[x,y,z] → [x,-z,y]`, followed by GMR
  retargeting and a ground offset. Pico pelvis and G1 root positions are not directly comparable.
- World coordinates refer to the recording's tracking origin; sessions are not automatically
  aligned. Recenter or relocalization can cause jumps that the validity mask does not detect.
- `pico_head_pose` describes the headset. Camera intrinsics and extrinsics are stored in
  `meta/pico_ego_release.json`.
- Body joint 0 is the pelvis. Hand joint 0 is the palm; **joint 1 is the wrist**.
  See the [pipeline reference](pipeline_reference.md#raw-data) for the full body joint order.

For local coordinates, given anchor `(p_a,q_a)` and joint `(p_j,q_j)`:

```text
p_local = R(q_a)^T · (p_j - p_a)
q_local = inverse(q_a) · q_j
```

Use the pelvis as the body anchor and each wrist as its hand anchor. Resample in time
before computing local poses, and handle invalid poses using the masks first. Local poses
omit the anchor's world trajectory and cannot reconstruct world poses without it.

## Timing and missing data

- The camera's first-frame `timeStampNs` defines time zero. Web annotations use the video timeline.
- Video, Pico poses, and G1 trajectories select the nearest source frame at each output time.
  Ties select the earlier frame; no interpolation is applied.
- G1 pkl files require `times`, `time_origin_ns`, and
  `time_source=camera_first_frame_timestamp_ns`, with the same origin as the tracking data.
- Missing hands trigger export confirmation. Gaps of at most 0.25 seconds, bounded by
  tracked observations, use the last valid pose. Other missing hand poses use all-zero
  placeholders. Originally invalid points keep `joint_valid=0`.
- Missing head/body poses use zero position and an identity quaternion. Numerically valid
  but inactive hand observations are retained: training must check both `hand_active` and
  `joint_valid`. Zero-quaternion placeholders are not valid rotations.
- Quality records are saved in the sidecar. Invalid poses do not automatically discard an episode.

## Camera metadata and sidecar

`meta/pico_ego_release.json` records the schema, sampling policy, coordinate conventions,
camera parameters, time alignment, and quality checks. Physical left/right eye assignment,
per-eye intrinsic interpretation, and extrinsic direction remain marked `UNVERIFIED`.
Verify them with device-specific occlusion tests and calibration before geometric reconstruction;
left/right image field names alone do not establish physical calibration.

## Loading example

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "pico_ego/SESSION", root="lerobot_v3/SESSION", video_backend="pyav"
)
sample = ds[0]
body = sample["observation.pico_body_pose"].reshape(24, 7)
print(body[0, :3])                        # Pico pelvis world xyz
print(sample["observation.g1_qpos"].shape)  # (36,)
print(sample["task"])
```

The default TorchCodec backend needs compatible FFmpeg libraries; this example uses PyAV.
