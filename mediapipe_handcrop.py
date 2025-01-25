import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import os

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import os


class MediapipeHandCrop:
    """
    MediapipeHandCrop

    A class to detect and crop hands from images using MediaPipe.

    Parameters:
        max_num_hands (int): Maximum number of hands to detect. Default is 1.
        min_detection_confidence (float): Minimum confidence for hand detection. Default is 0.5.
        include_characteristic_vectors (bool): Whether to include normalized characteristic vectors. Default is False.
        target_size (tuple): Target size for scaling characteristic vectors. Default is (64, 64).

    Methods:
        __call__(image, return_as_numpy=False): Detect and crop hand from the given image.

    Returns:
        A tuple containing:
        - Cropped hand image (PIL.Image or numpy.ndarray).
        - Characteristic vectors (list of scaled landmarks), if enabled.
    """

    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, include_characteristic_vectors=False, target_size=(64, 64)):
        self.include_characteristic_vectors = include_characteristic_vectors
        self.target_size = target_size
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )

    def __call__(self, image):
        # Handle image input formats
        if isinstance(image, Image.Image):
            image = np.array(image)  # Convert PIL to NumPy array
            print(f"converted image = {image}")
        elif isinstance(image, np.ndarray):
            pass  # Image is already a NumPy array
        elif isinstance(image, str) and os.path.exists(image):
            image = cv2.imread(image)  # Read image from file path
        else:
            raise ValueError("Input should be a PIL.Image, numpy.ndarray, or a valid file path string.")

        # Convert to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = self.mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:

            # Process detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                height, width, _ = image.shape
                x_coords = [int(landmark.x * width) for landmark in hand_landmarks.landmark]
                y_coords = [int(landmark.y * height) for landmark in hand_landmarks.landmark]

                # Calculate bounding box
                x_min, x_max = max(0, min(x_coords)), min(width, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(height, max(y_coords))

                # Ensure the bounding box is a square
                box_width = x_max - x_min
                box_height = y_max - y_min
                max_side = max(box_width, box_height)

                # Adjust bounding box to make it square
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                half_side = max_side // 2

                x_min_square = max(0, x_center - half_side)
                x_max_square = min(width, x_center + half_side)
                y_min_square = max(0, y_center - half_side)
                y_max_square = min(height, y_center + half_side)

                # Crop the hand region
                cropped_hand = image[y_min_square:y_max_square, x_min_square:x_max_square]

                # Process characteristic vectors if enabled
                if self.include_characteristic_vectors:
                    characteristic_vectors = []
                    cropped_width = x_max_square - x_min_square
                    cropped_height = y_max_square - y_min_square

                    for landmark in hand_landmarks.landmark:
                        # Recalculate the coordinates for the cropped image
                        norm_x = (landmark.x * width - x_min_square) / cropped_width
                        norm_y = (landmark.y * height - y_min_square) / cropped_height
                        scaled_x = norm_x * self.target_size[0]
                        scaled_y = norm_y * self.target_size[1]
                        characteristic_vectors.append([scaled_x, scaled_y])

                    return Image.fromarray(cropped_hand), characteristic_vectors  # RETURNS cropped image as Image object not np.array

                return Image.fromarray(cropped_hand)

        # If no hands were detected (should not reach here due to earlier check)
        return None



import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import os

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import os


class MediapipeHandCrop2:

    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, include_characteristic_vectors=False, target_size=(64, 64)):
        self.include_characteristic_vectors = include_characteristic_vectors
        self.target_size = target_size
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )

    def __call__(self, image):
        # Handle image input formats
        if isinstance(image, Image.Image):
            image = np.array(image)  # Convert PIL to NumPy array
        elif isinstance(image, np.ndarray):
            pass  # Image is already a NumPy array
        elif isinstance(image, str) and os.path.exists(image):
            image = cv2.imread(image)  # Read image from file path
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Input should be a PIL.Image, numpy.ndarray, or a valid file path string.")

        # Process the image with MediaPipe Hands
        results = self.mp_hands.process(image)

        if results.multi_hand_landmarks:

            # Process detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                height, width, _ = image.shape
                x_coords = [int(landmark.x * width) for landmark in hand_landmarks.landmark]
                y_coords = [int(landmark.y * height) for landmark in hand_landmarks.landmark]

                # Calculate bounding box
                x_min, x_max = max(0, min(x_coords)), min(width, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(height, max(y_coords))

                # Ensure the bounding box is a square
                box_width = x_max - x_min
                box_height = y_max - y_min
                max_side = max(box_width, box_height)

                # Adjust bounding box to make it square
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                half_side = max_side // 2

                x_min_square = max(0, x_center - half_side)
                x_max_square = min(width, x_center + half_side)
                y_min_square = max(0, y_center - half_side)
                y_max_square = min(height, y_center + half_side)

                # Crop the hand region
                cropped_hand = image[y_min_square:y_max_square, x_min_square:x_max_square]

                # Process characteristic vectors if enabled
                if self.include_characteristic_vectors:
                    characteristic_vectors = []
                    cropped_width = x_max_square - x_min_square
                    cropped_height = y_max_square - y_min_square

                    for landmark in hand_landmarks.landmark:
                        # Recalculate the coordinates for the cropped image
                        norm_x = (landmark.x * width - x_min_square) / cropped_width
                        norm_y = (landmark.y * height - y_min_square) / cropped_height
                        scaled_x = norm_x * self.target_size[0]
                        scaled_y = norm_y * self.target_size[1]
                        characteristic_vectors.append([scaled_x, scaled_y])

                    return Image.fromarray(cropped_hand), characteristic_vectors  # RETURNS cropped image as Image object not np.array

                return Image.fromarray(cropped_hand)

        # If no hands were detected (should not reach here due to earlier check)
        return None

