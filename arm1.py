import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,              # <<<--- THIS IS THE MAIN CHANGE
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Landmark IDs ---
THUMB_TIP_ID = 4
INDEX_FINGER_TIP_ID = 8

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip and convert BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Process the image and find hands
    results = hands.process(image)

    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get frame dimensions
    h, w, _ = image.shape

    # --- Calculation Logic for TWO hands ---
    if results.multi_hand_landmarks:
        # Loop through each detected hand
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            # --- Get Hand Label (Left or Right) ---
            # This 'handedness' object corresponds by index to the 'hand_landmarks'
            hand_info = results.multi_handedness[idx]
            hand_label = hand_info.classification[0].label # 'Left' or 'Right'
            
            # 1. Get landmarks for the current hand
            thumb_tip = hand_landmarks.landmark[THUMB_TIP_ID]
            index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP_ID]

            # 2. Convert to pixel coordinates
            thumb_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_px = (int(index_tip.x * w), int(index_tip.y * h))

            # 3. Calculate Euclidean distance
            distance = int(math.hypot(index_px[0] - thumb_px[0], index_px[1] - thumb_px[1]))

            # --- Visualization for the current hand ---
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
                
            # Draw line between tips
            cv2.line(image, thumb_px, index_px, (0, 255, 0), 3)

            # Draw circles at tips
            cv2.circle(image, thumb_px, 10, (0, 0, 255), -1)
            cv2.circle(image, index_px, 10, (0, 0, 255), -1)

            # --- Display Text (position based on L/R) ---
            display_text = f'{hand_label} Hand: {distance} px'
            
            if hand_label == 'Left':
                text_pos = (50, 50) # Top-left
            else: # Right
                text_pos = (w - 450, 50) # Top-right (adjusted for text length)
                
            cv2.putText(
                image,
                display_text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )

    # Show the final image
    cv2.imshow('Two-Hand Distance Measurement', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()