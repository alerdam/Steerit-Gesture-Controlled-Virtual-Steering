import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os



# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle between three points.
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0  # Handle zero magnitude case
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0) 
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def detect_ruler_sin_angle(frame):
    """Detect horizontal ruler angle and return sin(angle)."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_landmarks_list = []

    # Check if two hands are detected
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get all hand landmark positions
            landmarks = [(int(hand_landmarks.landmark[i].x * frame.shape[1]),
                          int(hand_landmarks.landmark[i].y * frame.shape[0])) for i in range(21)]
            hand_landmarks_list.append(landmarks)

    # If two hands detected, find bounding box covering both hands
    if len(hand_landmarks_list) == 2:
        x_coords = [x for hand in hand_landmarks_list for x, _ in hand]
        y_coords = [y for hand in hand_landmarks_list for _, y in hand]

        x_min, x_max = max(0, min(x_coords) - 50), min(frame.shape[1], max(x_coords) + 50)  # Increased margin
        y_min, y_max = max(0, min(y_coords) - 50), min(frame.shape[0], max(y_coords) + 50)  # Increased margin


        # Extract ROI (Region of Interest)
        roi = frame[y_min:y_max, x_min:x_max]

        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return None

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=3)
        if lines is not None:
            best_line = None
            best_length = 0
            for line in lines:
                x1r, y1r, x2r, y2r = line[0]
                line_length = np.sqrt((x2r - x1r)**2 + (y2r - y1r)**2)
                if line_length > best_length:
                    best_length = line_length
                    best_line = line[0]
            if best_line is not None:
                x1r, y1r, x2r, y2r = best_line
                angle_horizontal = np.degrees(np.arctan2(y2r - y1r, x2r - x1r))
                sin_angle = np.round(np.sin(np.radians(angle_horizontal)),2)


                # Draw detected ruler line
                cv2.line(roi, (x1r, y1r), (x2r, y2r), (0, 255, 0), 2)
                cv2.putText(frame, f"Sin(Angle): {sin_angle:.2f}", (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"framle label : {sin_angle}")
                return sin_angle

    return None



# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Initialize Matplotlib 3D plot
fig = plt.figure("Record Session",figsize=(7, 3))
ax = fig.add_subplot(121, projection='3d')
ax_img = fig.add_subplot(122)

# Initialize arrays to store hand landmarks
PLOT_hand_landmarks_3d = [np.zeros((21, 3)), np.zeros((21, 3))]

# Initialize data list to store hand landmarks for .npy
all_hand_data = []
label_data = []

# Function to update the 3D plot and save data
def update(frame_num):
    global PLOT_hand_landmarks_3d
    global all_hand_data

    ret, image = cap.read()
    if not ret:
        return ax, ax_img

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Plot setup
    ax.clear()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=-90, azim=-90)
    ax_img.clear()
    ax_img.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax_img.axis('off')


    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        PLOT_hand_landmarks_3d_temp = [[], []]
        frame_data = []
        sin_angle = detect_ruler_sin_angle(image)
        if sin_angle is not None:
            label_data.append(sin_angle)
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_data = []
                for landmark_index, landmark in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    lx = abs(float(f"{landmark.x :.2f}")- 1) # With mirror vision
                    ly = float(f"{landmark.y :.2f}")
                    lz = float(f"{landmark.z:.2f}")
                    PLOT_hand_landmarks_3d_temp[hand_index].append([lx, ly, lz])
                    hand_data.append([lx, ly, lz])
                PLOT_hand_landmarks_3d[hand_index] = np.array(PLOT_hand_landmarks_3d_temp[hand_index])
                frame_data.append(hand_data)

            all_hand_data.append(frame_data)

            for hand_index in range(2):
                ax.scatter(PLOT_hand_landmarks_3d[hand_index][:, 0], 
                        PLOT_hand_landmarks_3d[hand_index][:, 1],
                        PLOT_hand_landmarks_3d[hand_index][:, 2], c='red')

    ax_img.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return ax, ax_img

# Animate the 3D plot and camera feed
ani = animation.FuncAnimation(fig, update, interval=60, blit=False)

plt.show()

# Release resources
cap.release()
cv2.destroyAllWindows()

# Convert and save data
if all_hand_data:
    all_hand_data_np = np.array(all_hand_data)
    label_data_np = np.array(label_data)
    np.save('DATA/landmark_positions.npy', all_hand_data_np)
    np.save("DATA/label_data.npy", label_data)
    print("Hand landmark data saved to DATA/landmark_positions.npy")
    print("Direksiyon data saved to DATA/label_data.npy")
    print("Shape of hand_data (npy):", all_hand_data_np.shape)
else:
    print("No hand landmark data captured.")
