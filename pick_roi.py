"""Click on a video frame to pick ROI and line points for main.py.

Usage:
    python pick_roi.py

1. Click 4 ROI corners (in order around the polygon).
2. Click 2 line endpoints.
3. Press any key when done — copy the printed lists into main.py.
"""

import cv2

VIDEO_PATH = "input_video/drones_swarm.mp4"
FRAME_SIZE = (1280, 720)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise SystemExit(f"Could not read frame from {VIDEO_PATH}")

frame = cv2.resize(frame, FRAME_SIZE)
display = frame.copy()
points = []


def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    points.append((x, y))
    cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
    if len(points) > 1:
        cv2.line(display, points[-2], points[-1], (0, 255, 0), 2)
    if len(points) == 4:
        cv2.line(display, points[3], points[0], (0, 255, 0), 2)
    cv2.imshow("pick points", display)
    print(points)


cv2.imshow("pick points", display)
cv2.setMouseCallback("pick points", on_click)
print("Click 4 ROI corners, then 2 line endpoints. Press any key when done.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print()
print("roi_points =", points[:4])
print("line_points =", points[4:6])
