# Pose Driven 3D Characters

Stand in front of the camera and a rigged 3D character copies your pose in real time.

![Fig.1: Example 1](https://raw.githubusercontent.com/MathematicLove/my-cv/main/content/projects/POSE_DRIVEN_3D_CHARACTERS/POSE_DRIVEN_3D_EXAMPLE_1.png)

![Fig.2: Example 2](https://raw.githubusercontent.com/MathematicLove/my-cv/main/content/projects/POSE_DRIVEN_3D_CHARACTERS/POSE_DRIVEN_3D_EXAMPLE_2.png)

## Model

Two models, no training:

- **YOLO pose** (`models/yolo26s-pose.pt`) returns 17 body keypoints per person
  (shoulders, elbows, wrists, hips, knees, ankles, face), tracked across frames and
  smoothed with a one euro filter.
- **Rigged glTF characters** (`models/characters/*.glb`) carry their own skeletons:
  `Michelle` (Mixamo), `Rigged Figure` and `Cesium Man` (Khronos samples). Switch with
  `M` / `N`. The `.glb` files are read directly - meshes, skinning weights, bones and
  embedded textures - and drawn by a small software rasterizer, so there is no
  game engine and no GPU requirement.

## Logic

Each bone is aimed at the direction its keypoints describe. A bone points along
`parent.rotation @ child_offset`, so making that equal the target gives its local
rotation:

```
R = animated_parent.rotation^-1 @ swing(rest -> target) @ bind_parent.rotation
```

Both accumulated rotations are needed, animated **and** bind. Using the animated one on
both sides only works when the bind rotation is identity, which is never true once a rig
has an axis-conversion root, and it misaims every limb by up to 20 degrees.

The rest of it:

- **Depth.** One camera sees no depth, so a limb shorter than its own reference length
  must be angled toward or away from the lens; that missing length is recovered as a
  damped Z component.
- **Turning.** Shoulders narrower than the body's proportions mean the torso is turned.
  That yaw is folded into every bone's swing, because each driven bone's orientation is
  absolute - spinning one alone is undone by the next one down.
- **Mirroring.** The camera image is flipped, so left and right chains swap and the
  horizontal axis flips: raise your right arm and the character raises its right arm,
  facing you.
- **Speed.** All per-triangle work - projection, culling, lighting, colour - is batched
  in numpy, leaving one OpenCV polygon fill per triangle, and dense meshes are thinned
  at load to a triangle budget. Roughly 30-70 FPS depending on the character.

Views cycle with `K`: character only, character with a camera inset, or camera with a
character inset. Keypoints and the skeleton are drawn on the camera image.

## Image

- Tag: flugmaschine/pose-driven-3d-characters:latest
- Platform: linux/amd64
- Base: python:3.12-slim

## Run

This is a camera and GUI application. It needs a Linux host with a webcam and an
X11 display. It cannot reach the camera or screen on Docker Desktop for macOS or
Windows.

```bash
xhost +local:docker

docker run --rm \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network host \
  flugmaschine/pose-driven-3d-characters:latest
```

Or with docker compose:

```bash
docker compose up
```

Pass CLI flags through to `src/app.py` after the image name, for example:

```bash
docker run --rm \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network host \
  flugmaschine/pose-driven-3d-characters:latest \
  python src/app.py --model "Rigged Figure" --camera 1 --pose-size 448 --no-mirror
```

## Controls

Keys: `K` view, `M` / `N` model, `F` detail, `L` keypoint names, `Y` / `P` rotate view,
`R` reset view, `D` debug, `H` help, `Q` quit.
