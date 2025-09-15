import numpy as np
import math


def calculate_3d_distance_manual(point1, point2):

    x1, y1, z1 = point1
    x2, y2, z2 = point2

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return distance

def extract_palm_landmarks(hand_pair_array):
    """Extracts palm landmarks (0, 1, 5, 9, 13) from a hand pair NumPy array."""
    palm_landmarks_indices = [0, 1, 5, 9, 13]
    palm_landmarks_array = np.zeros((2, 5, 3))
    if hand_pair_array is not None and len(hand_pair_array) == 2:
        for hand_index in range(2):
            if isinstance(hand_pair_array[hand_index], np.ndarray) and hand_pair_array[hand_index].shape == (21, 3):
                for i, palm_index in enumerate(palm_landmarks_indices):
                    palm_landmarks_array[hand_index, i] = hand_pair_array[hand_index][palm_index]
    return palm_landmarks_array

def calculate_distance_vector(point1, point2):
    """Calculates the distance vector between two 3D points, rounded to 2 decimal places."""
    distance_vector = point2 - point1
    return np.round(distance_vector, 2)

def calculate_slope(distance_vector):
    """Calculates the slope (y/x) from a distance vector, rounded to two decimal places."""
    if distance_vector[0] == 0:
        return float('inf')  # Handles division by zero, using infinity for vertical lines
    return round(distance_vector[1] / distance_vector[0], 2)