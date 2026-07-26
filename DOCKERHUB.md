# Pose Driven 3D Characters

Stand in front of the camera and a rigged 3D character copies your pose in real time.
Pose detection uses YOLO pose. Rigged glTF characters are read directly and drawn by a
small software rasterizer - no game engine, no GPU requirement.

## Image

- Tag: flugmaschine/pose-driven-3d-characters:latest
- Platform: linux/amd64
- Base: python:3.12-slim

## Run

This is a camera and GUI application. It needs a Linux host with a webcam and an
X11 display. It cannot reach the camera or screen on Docker Desktop for macOS or
Windows.

```
xhost +local:docker

docker run --rm \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network host \
  flugmaschine/pose-driven-3d-characters:latest
```

## Controls

- `K` view, `M` / `N` model, `F` detail, `L` keypoint names, `Y` / `P` rotate view,
  `R` reset view, `D` debug, `H` help, `Q` quit.
