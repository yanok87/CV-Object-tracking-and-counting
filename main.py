# ----------------------------
# STEP 1 : IMPORT LIBRARIES
# ----------------------------
import os
from ultralytics import YOLO  # YOLO object detection & tracking
import cv2  # OpenCV for video handling & drawing
import numpy as np  # Numerical computationsQ
from collections import defaultdict, deque  # Memory structures for tracking

# ----------------------------
# STEP 2 : LOAD YOLO MODEL & VIDEO
# ----------------------------
model = YOLO("best.pt")  # Load YOLO model weights
cap = cv2.VideoCapture("input_video/drones_swarm.mp4")  # Load input video

FRAME_SIZE = (1280, 720)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
os.makedirs("output_video", exist_ok=True)
out = cv2.VideoWriter(
    "output_video/drones_swarm_counted.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    FRAME_SIZE,
)

# # ---------------------------------------------------------
# STEP 3 : MEMORY FOR CENTER SMOOTHING
# ---------------------------------------------------------
# Stores last 10 center points for each tracked object
center_history = defaultdict(lambda: deque(maxlen=10))

# # Stores previous center for line-crossing comparison
previous_centers = {}


# ---------------------------------------------------------
# STEP 4 : CENTER SMOOTHING FUNCTION
# ---------------------------------------------------------
def get_smooth_center(track_id, new_center):
    # Add current center to history
    center_history[track_id].append(new_center)

    # Separate X and Y coordinates
    xs = [c[0] for c in center_history[track_id]]
    ys = [c[1] for c in center_history[track_id]]

    # Compute average X and Y
    avg_x = int(sum(xs) / len(xs))
    avg_y = int(sum(ys) / len(ys))

    # Return smoothed center
    return (avg_x, avg_y)


# ---------------------------------------------------------
# STEP 5 : LINE CROSSING FUNCTION
# ---------------------------------------------------------
def is_center_crossing_line(prev_center, curr_center, line_p1, line_p2):
    x1, y1 = line_p1
    x2, y2 = line_p2

    px, py = prev_center
    cx, cy = curr_center

    # Determine which side of the line the points lie on
    prev_side = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    curr_side = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)

    # If sign changes → crossing detected
    return prev_side * curr_side < 0


# ---------------------------------------------------------
# STEP 6 : ROI & LINE DRAWING FUNCTIONS
# ---------------------------------------------------------
def draw_roi(frame, points, color):
    # Create a copy of the original frame
    # This is done so we can draw on the copy (overlay)
    # without permanently modifying the original frame
    overlay = frame.copy()

    # Fill the polygon defined by 'points' on the overlay image
    # np.array(points, np.int32) converts the list of points
    # into the format required by OpenCV
    # The polygon is filled with the given color
    cv2.fillPoly(overlay, [np.array(points, np.int32)], color)

    # Blend the overlay with the original frame
    # 0.25 → transparency of the overlay (ROI color)
    # 0.75 → transparency of the original frame
    # 0    → gamma value (brightness adjustment)
    # This creates a semi-transparent ROI effect
    return cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)


def draw_line(frame, line_points):
    cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 4)
    return frame


# # ---------------------------------------------------------
# # STEP 7 : COUNTING MEMORY
# # ---------------------------------------------------------
# # It stores ONLY unique values (no duplicate IDs allowed)
# # Example:
# # counted_down_ids = {3, 7, 12}
# # If track_id = 7 comes again → already counted → skip counting
counted_ids = set()

# ---------------------------------------------------------
# STEP 9 : DEFINE ROIs & COUNTING LINES
# ---------------------------------------------------------
# Run: python pick_roi.py  → paste printed lists here
roi_points = [(15, 3), (17, 714), (1261, 714), (1262, 3)]
line_points = [(118, 2), (113, 713)]

# ---------------------------------------------------------
# STEP 10 : INITIALIZE COUNTERS
# ---------------------------------------------------------
drone_count = 0

# ============================================================
# STEP 11 : MAIN VIDEO LOOP
# ============================================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, FRAME_SIZE)

    # Mask ROI
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([roi_points], dtype=np.int32)], 255)

    # Apply a bitwise AND operation on the frame
    # First 'frame'  → source image (input image)
    # Second 'frame' → same image used again so pixel values remain unchanged
    # Using the same image twice makes this operation act like masking, not comparison
    # mask=mask → only pixels where mask is white (255) are kept
    # Pixels where mask is black (0) are removed (set to black)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # YOLO detection + tracking
    # Run the model on the masked frame to detect and track objects
    # model.track() performs object detection + tracking across frames
    # masked_frame → input image with only the ROI visible
    # persist=True → keeps tracking IDs consistent across frames
    # verbose=False → disables console/log output
    # [0] → extracts the first result from the returned list (current frame result)
    results = model.track(masked_frame, persist=True, verbose=False)[0]

    # Draw ROIs
    frame = draw_roi(frame, roi_points, (255, 0, 0))

    # Draw counting lines
    frame = draw_line(frame, line_points)

    for box in results.boxes:

        # Skip detections without tracking ID
        if box.id is None:
            continue

        track_id = int(box.id)

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Compute raw center
        raw_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Smooth center
        smooth_center = get_smooth_center(track_id, raw_center)

        # ----------------------------
        # DOWN COUNT
        # ----------------------------
        # Check if this object existed in the previous frame
        # (needed to compare movement across frames)
        if track_id in previous_centers and track_id not in counted_ids:

            # Check whether the object’s center crossed the DOWN counting line
            # between the previous frame and the current frame
            if is_center_crossing_line(
                    previous_centers[track_id],  # center position in previous frame
                    smooth_center,  # center position in current frame
                    line_points[0],  # starting point of DOWN line
                    line_points[1]  # ending point of DOWN line
            ):
                # Increment the DOWN counter by 1
                # because a valid line crossing is detected
                drone_count += 1

                # Add this track_id to the counted_down_ids set
                # to ensure the same object is NOT counted again
                counted_ids.add(track_id)

        # Update previous center
        previous_centers[track_id] = smooth_center

        # Draw **only center** and ID
        # Green = not yet counted, Red = counted
        color = (0, 255, 0) if track_id not in counted_ids else (0, 0, 255)
        cv2.circle(frame, smooth_center, 5, color, -1)
        cv2.putText(frame, f"{track_id}", (smooth_center[0] + 5, smooth_center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ---------------------------------------------------------
    # STEP 12 : DISPLAY COUNTS
    # ---------------------------------------------------------
    cv2.rectangle(frame, (5, 5), (220, 55), (0, 255, 255), -1)
    cv2.putText(frame, f"Count : {drone_count}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 4)

    out.write(frame)
    cv2.imshow("Object Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# STEP 13 : CLEANUP
# ============================================================
cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved: output_video/drones_swarm_counted.mp4")