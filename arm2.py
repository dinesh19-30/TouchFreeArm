import cv2
import mediapipe as mp
import math
import time 

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils 
mp_hands = mp.solutions.hands 

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Landmark IDs ---
THUMB_TIP_ID = 4
INDEX_FINGER_TIP_ID = 8
LITTLE_FINGER_TIP_ID = 20

# --- UI Properties (THE CHANGES ARE HERE) ---
UI_MARGIN = 15              
BOX_WIDTH = 30              
BOX_HEIGHT = 30             
BOX_SPACING = 40            # <<<--- CHANGED (from 25)
COLOR_INACTIVE = (80, 80, 80)
COLOR_ACTIVE = (0, 255, 0)
COLOR_HOVER = (255, 255, 0)
TOGGLE_COOLDOWN = 0.5

# --- State Variables ---
left_box_states = [False] * 5
right_box_states = [False] * 5
left_box_last_toggle = [0.0] * 5
right_box_last_toggle = [0.0] * 5

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Full Screen Setup
window_name = 'Hand-Controlled Servo UI'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
        
    current_time = time.time() 
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, _ = image.shape

    # --- Hand processing loop ---
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            hand_info = results.multi_handedness[idx]
            hand_label = hand_info.classification[0].label
            
            # Get landmarks
            thumb_tip = hand_landmarks.landmark[THUMB_TIP_ID]
            index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP_ID]
            little_tip = hand_landmarks.landmark[LITTLE_FINGER_TIP_ID] 

            # Convert to pixels
            thumb_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_px = (int(index_tip.x * w), int(index_tip.y * h))
            little_px = (int(little_tip.x * w), int(little_tip.y * h)) 

            # Calculate distance
            distance = int(math.hypot(index_px[0] - thumb_px[0], index_px[1] - thumb_px[1]))

            # Draw Landmarks
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.line(image, thumb_px, index_px, (0, 255, 0), 3)
            cv2.circle(image, little_px, 10, COLOR_HOVER, 2)

            # --- Text and Box properties ---
            display_text = f'{hand_label} Hand: {distance} px'
            font_scale = 0.7
            font_thickness = 2
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            
            # --- Decoupled UI Placement ---
            
            if hand_label == 'Left':
                text_pos = (UI_MARGIN, UI_MARGIN)
                box_start_x = UI_MARGIN
                box_start_y = UI_MARGIN + 40 # 40px below text
                box_states = left_box_states
                last_toggle_times = left_box_last_toggle
                
            else: # Right
                text_size, _ = cv2.getTextSize(display_text, font_face, font_scale, font_thickness)
                text_width = text_size[0]
                text_pos = (w - text_width - UI_MARGIN, UI_MARGIN)
                box_start_x = w - BOX_WIDTH - UI_MARGIN
                box_start_y = UI_MARGIN + 40
                box_states = right_box_states
                last_toggle_times = right_box_last_toggle
                
            # Draw Text
            cv2.putText(
                image, display_text, text_pos, font_face, 
                font_scale, (0, 255, 0), font_thickness
            )
            
            # --- Draw 5 Boxes ---
            for i in range(5):
                # Calculate box position
                box_x1 = box_start_x
                box_x2 = box_start_x + BOX_WIDTH
                box_y1 = box_start_y + i * (BOX_HEIGHT + BOX_SPACING) 
                box_y2 = box_y1 + BOX_HEIGHT

                # Check hover
                is_hovering = (box_x1 < little_px[0] < box_x2) and \
                              (box_y1 < little_px[1] < box_y2)
                              
                current_color = COLOR_INACTIVE

                # Check toggle
                if is_hovering and (current_time - last_toggle_times[i] > TOGGLE_COOLDOWN):
                    box_states[i] = not box_states[i] 
                    last_toggle_times[i] = current_time 
                
                # Set color
                if box_states[i]: 
                    current_color = COLOR_ACTIVE
                elif is_hovering:
                    current_color = COLOR_HOVER

                # Draw box
                cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), 
                              current_color, -1) 
                
                # Draw the number
                label_text = f"{i+1}"
                cv2.putText(image, label_text, (box_x1 + 8, box_y1 + 23), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()