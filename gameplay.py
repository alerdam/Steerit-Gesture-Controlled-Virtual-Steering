import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
from tensorflow import keras
import handlib


# Load the model
modelpath = f"M_timeaware.keras"
model = keras.models.load_model(modelpath)

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize Matplotlib 3D plot
fig = plt.figure("Model Predictions",figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
# Initialize arrays to store hand landmarks
PLOT_hand_landmarks_3d = [np.zeros((21, 3)), np.zeros((21, 3))]
distvector_ConnectionLines = np.zeros((5, 3))
slopes_ConnectionLines = np.zeros((5, 1))  # Ensure correct shape for slopes
slopes_KnuckleLines = np.zeros(2)
maxgasval = None
mingasval = None

# Initialize data list to store hand landmarks for .npy
palm_landmarks_indices = [0, 1, 5, 9, 13]

def update(notimportant):
    global PLOT_hand_landmarks_3d
    global slopes_ConnectionLines
    global slopes_KnuckleLines
    global maxgasval, mingasval

    ret, image = cap.read()
    if not ret:
        return ax
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

    input_data = np.array([]) # Initialize input_data
    scaled_gas_value = 0 # Initialize scaled_gas_value here

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        PLOT_hand_landmarks_3d_temp = [[], []]
        frame_data = [[None] * 21, [None] * 21] # Initialize with None
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            for landmark_index, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                lx = abs(float(f"{landmark.x :.2f}") - 1)  # With mirror vision
                ly = float(f"{landmark.y :.2f}")
                lz = float(f"{landmark.z:.2f}")
                PLOT_hand_landmarks_3d_temp[hand_index].append([lx, ly, lz])
                frame_data[hand_index][landmark_index] = [lx, ly, lz]
            PLOT_hand_landmarks_3d[hand_index] = np.array(PLOT_hand_landmarks_3d_temp[hand_index])

        palm_array = handlib.extract_palm_landmarks(PLOT_hand_landmarks_3d)

        for palm_index in range(len(palm_landmarks_indices)):
            LeftPalm = palm_array[0, palm_index]
            RightPalm = palm_array[1, palm_index]
            distance_vector = handlib.calculate_distance_vector(LeftPalm, RightPalm)
            distvector_ConnectionLines[palm_index] = distance_vector
            slope = handlib.calculate_slope(distance_vector)
            slopes_ConnectionLines[palm_index, 0] = slope

        # Knuckle lines
        if all(frame_data[0][i] is not None for i in [5, 17]) and all(frame_data[1][i] is not None for i in [5, 17]):
            landmark5_hand0 = np.array(frame_data[0][5])
            landmark17_hand0 = np.array(frame_data[0][17])
            distance_vector_hand0 = handlib.calculate_distance_vector(landmark5_hand0, landmark17_hand0)
            slopes_KnuckleLines[0] = handlib.calculate_slope(distance_vector_hand0)

            landmark5_hand1 = np.array(frame_data[1][5])
            landmark17_hand1 = np.array(frame_data[1][17])
            distance_vector_hand1 = handlib.calculate_distance_vector(landmark5_hand1, landmark17_hand1)
            slopes_KnuckleLines[1] = handlib.calculate_slope(distance_vector_hand1)

        # Calculate y_diff
        y_diff = None # Initialize y_diff
        if frame_data[0][0] is not None and frame_data[1][0] is not None:
            hand0_landmark0 = np.array(frame_data[0][0])
            hand1_landmark0 = np.array(frame_data[1][0])
            y_diff = hand0_landmark0[1] - hand1_landmark0[1]

        # Create input_data if all components are available
        if slopes_ConnectionLines.size > 0 and slopes_KnuckleLines.size == 2 and y_diff is not None:
            # Flatten the slope arrays and then stack them as a row
            flattened_slopes_connection = slopes_ConnectionLines.flatten()
            flattened_slopes_knuckle = slopes_KnuckleLines.flatten()
            input_data = np.hstack((flattened_slopes_connection, flattened_slopes_knuckle, np.array([y_diff])))
            input_data = input_data.reshape(1, -1)

            # Model output
            prediction = model.predict(input_data, verbose=0)[0][0]
            prediction = round(prediction, 2)

            # Gas
            fingertips_right_hand = [] # List to hold individual fingertip landmark positions
            fingertip_indices = [8, 12, 16, 20] # Index, Middle, Ring, Pinky fingertips
            hand0_landmark1 = np.array(frame_data[0][1])
            hand1_landmark1 = np.array(frame_data[1][1])
            for tip_idx in fingertip_indices:
                fingertips_right_hand.append(np.array(frame_data[1][tip_idx]))
            # Convert list of arrays to a single NumPy array
            fingertips_right_hand_array = np.array(fingertips_right_hand)
            # Calculate
            sumdistance = 0
            for ft_id in fingertips_right_hand_array:
                gas_distance = handlib.calculate_3d_distance_manual(ft_id, hand0_landmark1)
                gas_distance += handlib.calculate_3d_distance_manual(ft_id, hand1_landmark1)
                sumdistance += gas_distance
            # Max and Min Values of summation
            if maxgasval is None or maxgasval < sumdistance:
                maxgasval = sumdistance
            if mingasval is None or mingasval > sumdistance:
                mingasval = sumdistance

            # Calculate scaled_gas_value after updating maxgasval and mingasval
            if maxgasval is not None and mingasval is not None:
                try:
                    # Removed the premature return here
                    scaled_gas_value = (sumdistance - mingasval) / (maxgasval - mingasval)
                    if scaled_gas_value < 0.2:
                        scaled_gas_value = 0
                except ZeroDivisionError:
                    print("Error: Cannot divide by zero!")

        # Outputs
        print(" --------------------RESULTS--------------------------")
        print(f"Steering Value : {prediction:.2f}")
        print("---------------------------------------------------------")

        # Plot
        for hand_index in range(2):
            if PLOT_hand_landmarks_3d[hand_index].shape == (21, 3):
                ax.scatter(PLOT_hand_landmarks_3d[hand_index][:, 0],
                            PLOT_hand_landmarks_3d[hand_index][:, 1],
                            PLOT_hand_landmarks_3d[hand_index][:, 2], c=['red'])

    return ax

# Animate the 3D plot and camera feed 
ani = animation.FuncAnimation(fig, update, interval=30, blit=False) # Reduced interval for smoother animation

plt.show()

# Release resources
cap.release()
cv2.destroyAllWindows()