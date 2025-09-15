import numpy as np

def numpy_to_hand_pairs(npy_filename):
    """
    Reads a .npy file containing hand landmark data and converts it into a NumPy array,
    grouped by hand pairs (hand 0 and hand 1), with each hand's landmarks in order.
    """
    loaded_data = np.load(npy_filename)
    num_frames, num_hands, num_landmarks, coords = loaded_data.shape

    data = []
    for frame_index in range(num_frames):
        frame_data = [[], []]
        # Extract hand data for the frame
        hands = []
        for hand_index in range(num_hands):
            hand_landmarks = []
            for landmark_index in range(num_landmarks):
                x, y, z = loaded_data[frame_index, hand_index, landmark_index]
                hand_landmarks.append([x, y, z])
            hands.append((hand_landmarks[0][0], hand_landmarks))  # Store x value with landmarks
        # Sort hands so the one with the smaller x value is indexed as 0
        hands.sort(key=lambda h: h[0])
        # Assign sorted data to frame_data
        frame_data[0], frame_data[1] = hands[0][1], hands[1][1]
        data.append(frame_data)
    return np.array(data)

def extract_palm_landmarks(hand_pair_array):
    """Extracts palm landmarks (0, 1, 5, 9, 13) from a hand pair NumPy array."""
    num_frames = hand_pair_array.shape[0]
    palm_landmarks_indices = [0, 1, 5, 9, 13]
    palm_landmarks_array = np.zeros((num_frames, 2, 5, 3))
    for frame_index in range(num_frames):
        for hand_index in range(2):
            for i, palm_index in enumerate(palm_landmarks_indices):
                palm_landmarks_array[frame_index, hand_index, i] = hand_pair_array[frame_index, hand_index, palm_index]
    return palm_landmarks_array

def calculate_distance_vector(point1, point2):
    """Calculates the distance vector between two 3D points, rounded to 2 decimal places."""
    distance_vector = point2 - point1
    return np.round(distance_vector, 2)

def calculate_slope(distance_vector):
    """Calculates the slope (y/x) from a distance vector, rounded to two decimal places."""
    """Belki buraya geometrik dönüşüm!!!"""
    if distance_vector[0] == 0:
        return float('0')  # Handles division by zero
    return round(distance_vector[1] / distance_vector[0], 2)

# Get Landmarks to numpy:
npy_filename = 'DATA/landmark_positions.npy'
hand_pair_array = numpy_to_hand_pairs(npy_filename)

# Constants
num_frames = hand_pair_array.shape[0]
palm_landmarks_indices = [0, 1, 5, 9, 13]

if hand_pair_array.shape[0] > 0:  # Check hand_pair_array.

    # Special Points
    palm_array = extract_palm_landmarks(hand_pair_array)

    # Initialize etc.
    distvector_ConnectionLines = np.zeros((num_frames, len(palm_landmarks_indices), 3))
    slopes_ConnectionLines = np.zeros((num_frames, len(palm_landmarks_indices), 1))
    x_1, x_2 = slopes_ConnectionLines.shape[0], slopes_ConnectionLines.shape[1]
    slopes_ConnectionLines = slopes_ConnectionLines.reshape(x_1, x_2)
    distvector_KnuckleLines = np.zeros((num_frames, 2, 3))  # (frame, hand, [x,y,z])
    slopes_KnuckleLines = np.zeros((num_frames, 2))  # (frame, hand)
    y_diff_array = np.zeros(num_frames)

    # Connection lines of Palms
    for frame_index in range(num_frames):
        for palm_index in range(len(palm_landmarks_indices)):
            LeftPalm = palm_array[frame_index, 0, palm_index]
            RightPalm = palm_array[frame_index, 1, palm_index]
            distance_vector = calculate_distance_vector(LeftPalm, RightPalm)
            distvector_ConnectionLines[frame_index, palm_index] = distance_vector
            slope = calculate_slope(distance_vector)
            slopes_ConnectionLines[frame_index, palm_index] = slope

        # Knuckle lines
        for hand_index in range(2):
            landmark5 = hand_pair_array[frame_index, hand_index, 5]
            landmark17 = hand_pair_array[frame_index, hand_index, 17]
            distance_vector = calculate_distance_vector(np.array(landmark5), np.array(landmark17))
            slope = calculate_slope(distance_vector)
            slopes_KnuckleLines[frame_index, hand_index] = slope
            distvector_KnuckleLines[frame_index, hand_index] = distance_vector

        # Calculate y_diff
        hand0_landmark0 = hand_pair_array[frame_index, 0, 0]
        hand1_landmark0 = hand_pair_array[frame_index, 1, 0]
        y_diff = hand0_landmark0[1]  - hand1_landmark0[1]
        y_diff_array[frame_index] = np.round(y_diff, 2)

        # Load Label data
        labels_array = np.load("DATA\label_data.npy")


    # Save the numpy arrays to .npy files
    np.save('DATA/slopes_connection_lines.npy', slopes_ConnectionLines)
    np.save('DATA/slopes_knuckle_lines.npy', slopes_KnuckleLines)
    np.save('DATA/y_diff_array.npy', y_diff_array)

    # Console
    print(hand_pair_array.shape)

    selected_frame = 30
    print(f"Number of frames: {num_frames}")
    print(f"Hand pair array shape: {hand_pair_array.shape}")
    print(f"Palm landmark array shape: {palm_array.shape}")
    print(f"Connection Lines Distance vectors array shape: {distvector_ConnectionLines.shape}")
    print(f"Connection Lines Slopes array shape: {slopes_ConnectionLines.shape}")
    print(f"Knuckle Distance Vectors array shape: {distvector_KnuckleLines.shape}")
    print(f"Knuckle Slopes array shape: {slopes_KnuckleLines.shape}")
    print("")
    
    print("----------------------------------------------------------------------------")

    print(f"Looking at frame {selected_frame}")
    print(f"Distance vectors frame num {selected_frame} : \n{distvector_ConnectionLines[selected_frame]}")
    print(f"Slopes of frame num {selected_frame} : \n{slopes_ConnectionLines[selected_frame]}")

    print(f"Knuckle Distance vectors frame num {selected_frame} : \n{distvector_KnuckleLines[selected_frame]}")
    print(f"Knuckle Slopes frame num {selected_frame} : \n{slopes_KnuckleLines[selected_frame]}")

    print(f"y_diff frame num {selected_frame} : \n{y_diff}")

else:
    print("hand_pair_array is empty. No further processing.")