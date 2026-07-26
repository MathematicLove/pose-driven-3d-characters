# Pose Driven 3D Characters

Stand in front of the camera and a rigged 3D character copies your pose in real time.

<p align="center">
  <img src="https://github.com/MathematicLove/my-cv/blob/main/content/projects/POSE_DRIVEN_3D_CHARACTERS/POSE_DRIVEN_3D_EXAMPLE_1.png"
       width="400"
       height="300"
       alt="Fig.1: Example 1">
</p>

<p align="center">
  <img src="https://github.com/MathematicLove/my-cv/blob/main/content/projects/POSE_DRIVEN_3D_CHARACTERS/POSE_DRIVEN_3D_EXAMPLE_2.png"
       width="400"
       height="300"
       alt="Fig.2: Example 2">
</p>

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

Keys: `K` view, `M` / `N` model, `F` detail, `L` keypoint names, `Y` / `P` rotate view,
`R` reset view, `D` debug, `H` help, `Q` quit.

## Run

```bash
pip install -r requirements.txt
python src/app.py
```

```bash
python src/app.py --model "Rigged Figure"
python src/app.py --camera 1 --pose-size 448 --no-mirror
```

by Salimli Ayzek (Салимли Айзек): https://mathematiclove.github.io
