import cv2
import mediapipe as mp
import math
import numpy as np

# --- Global Variables for persistent objects ---
cap = None
hands = None
smoothed_dist = {'Left': None, 'Right': None}

# --- CONFIGURATION ---
DISTANCE_MIN = 20
DISTANCE_MAX = 200
ANGLE_MIN = 0
ANGLE_MAX = 180
SMOOTHING_FACTOR = 0.3

def initialize_tracker():
    """Initializes the Camera and MediaPipe Hands models."""
    global cap, hands, smoothed_dist
    
    # 1. Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # 2. Initialize OpenCV Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("PYTHON ERROR: Could not open webcam.")
        return False
        
    smoothed_dist = {'Left': None, 'Right': None}
    print("Python Tracker Initialized Successfully.")
    return True

def get_hand_angles():
    """Returns angles for L-Servo-1 and R-Servo-1."""
    global cap, hands, smoothed_dist
    
    angle_l_servo_1 = 90.0
    angle_r_servo_1 = 90.0
    
    if cap is None or not cap.isOpened():
        print("PYTHON ERROR: Camera not initialized.")
        return angle_l_servo_1, angle_r_servo_1

    success, image = cap.read()
    if not success:
        print("PYTHON ERROR: Ignoring empty camera frame.")
        return angle_l_servo_1, angle_r_servo_1

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    h, w, _ = image.shape
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx]
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'
            
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            # --- Calculate, Smooth, and Map Distance ---
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
            
            if smoothed_dist[hand_label] is None:
                smoothed_dist[hand_label] = distance
            else:
                smoothed_dist[hand_label] = (distance * SMOOTHING_FACTOR) + \
                                            (smoothed_dist[hand_label] * (1 - SMOOTHING_FACTOR))
            
            display_distance = smoothed_dist[hand_label]

            # Map pinch distance to servo angle (0-180)
            angle = np.interp(display_distance, 
                              [DISTANCE_MIN, DISTANCE_MAX], 
                              [ANGLE_MIN, ANGLE_MAX])
            
            if hand_label == 'Left':
                angle_l_servo_1 = float(angle)
            elif hand_label == 'Right':
                angle_r_servo_1 = float(angle)

    return angle_l_servo_1, angle_r_servo_1

def shutdown_tracker():
    """Releases the camera and closes models."""
    global cap, hands
    if cap:
        cap.release()
    if hands:
        hands.close()
    print("Python Tracker Shutdown.")

if __name__ == '__main__':
    # This block allows testing the Python script by itself
    if initialize_tracker():
        print("Testing tracker... Press 'q' to quit.")
        while True:
            l_angle, r_angle = get_hand_angles()
            print(f"Left Angle: {l_angle:.2f}, Right Angle: {r_angle:.2f}")
            
            ret, test_image = cap.read()
            if not ret:
                break
            cv2.imshow("Python Test Feed", cv2.flip(test_image, 1))
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        shutdown_tracker()
        cv2.destroyAllWindows()
