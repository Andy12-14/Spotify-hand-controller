import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math

# --- PyAutoGUI Commands for Spotify ---
# NOTE: The Spotify window must be the active window for these to work.
# These hotkeys are based on common Spotify desktop shortcuts.

# Playback
def play_pause():
    """Toggles play/pause on Spotify."""
    print("Action: Play/Pause")
    pyautogui.press('space')

def seek_forward():
    """Seeks forward in the current track."""
    print("Action: Seek Forward")
    pyautogui.hotkey('shift', 'right')

def seek_backward():
    """Seeks backward in the current track."""
    print("Action: Seek Backward")
    pyautogui.hotkey('shift', 'left')

def raise_volume():
    """Increases the system volume."""
    print("Action: Raise Volume")
    pyautogui.hotkey('ctrl', 'up')

def lower_volume():
    """Decreases the system volume."""
    print("Action: Lower Volume")
    pyautogui.hotkey('ctrl', 'down')

def decrease_playback_speed():
    """Decreases playback speed."""
    print("Action: Decrease Playback Speed")
    pyautogui.hotkey('shift', ',')

def increase_playback_speed():
    """Increases playback speed."""
    print("Action: Increase Playback Speed")
    pyautogui.hotkey('shift', '.')

def skip_to_previous():
    """Skips to the previous track."""
    print("Action: Skip to Previous")
    pyautogui.hotkey('ctrl', 'left')

def skip_to_next():
    """Skips to the next track."""
    print("Action: Skip to Next")
    pyautogui.hotkey('ctrl', 'right')

# General
def like_track():
    """Likes the current track."""
    print("Action: Like Track")
    pyautogui.hotkey('alt', 'shift', 'b')

def shuffle():
    """Toggles shuffle mode."""
    print("Action: Shuffle")
    pyautogui.hotkey('ctrl', 's')

def repeat():
    """Toggles repeat mode."""
    print("Action: Repeat")
    pyautogui.hotkey('ctrl', 'r')

def quick_search():
    """Opens the quick search bar."""
    print("Action: Quick Search")
    pyautogui.hotkey('ctrl', 'k')

def scroll_up():
    """Scrolls up in the active window."""
    # This will be continuous as long as the gesture is held.
    print("Action: Scroll Up")
    pyautogui.scroll(10)

def scroll_down():
    """Scrolls down in the active window."""
    # This will be continuous as long as the gesture is held.
    print("Action: Scroll Down")
    pyautogui.scroll(-10)

def return_home():
    """Simulates the hotkey to return home (or to the main UI)."""
    print("Action: Return Home")
    pyautogui.hotkey('alt', 'shift', 'h')

# --- Hand Tracking and Finger Counting Functions ---

def count_fingers_up(hand_landmarks):
    """
    Counts which fingers are up and returns a numpy array.
    Fingers are indexed from thumb (0) to pinky (4).
    A '1' means the finger is up, '0' means it's down.
    The list represents: [thumb, index, middle, ring, pinky].
    """
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]
    
    fingers = np.zeros(5, dtype=int)

    # For the thumb, check if the tip is to the right of the pip
    # This is a simple approximation for a thumb-up gesture.
    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x:
        fingers[0] = 1

    # For the other four fingers, check if the tip is above the pip
    for i in range(1, 5):
        if hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_pips[i]].y:
            fingers[i] = 1

    return fingers

def is_pinching(hand_landmarks, frame, threshold=25):
    """
    Detects a pinch gesture (thumb and index finger tips close together).
    Draws a line between the fingers to visualize the pinch distance.
    Returns True if a pinch is detected.
    """
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

    h, w, _ = frame.shape
    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

    distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
    
    cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)
    
    return distance < threshold, (thumb_x + index_x) // 2, (thumb_y + index_y) // 2

def get_gesture(fingers_up_array, current_state):
    """
    Maps finger positions to a specific gesture command based on the current state.
    """
    # Universal gesture to go back to the main menu
    if np.array_equal(fingers_up_array, [1, 0, 0, 0, 0]):
        return "return_home"

    if current_state == "PLAYBACK":
        if np.array_equal(fingers_up_array, [0, 0, 0, 0, 0]):
            return "play_pause"
        elif np.array_equal(fingers_up_array, [0, 1, 1, 0, 0]):
            return "raise_volume"
        elif np.array_equal(fingers_up_array, [0, 0, 0, 1, 1]):
            return "lower_volume"
        elif np.array_equal(fingers_up_array, [0, 0, 1, 1, 0]):
            return "seek_forward"
        elif np.array_equal(fingers_up_array, [0, 1, 0, 0, 0]):
            return "seek_backward"
        elif np.array_equal(fingers_up_array, [0, 1, 0, 1, 0]):
            return "decrease_playback_speed"
        elif np.array_equal(fingers_up_array, [0, 0, 1, 0, 1]):
            return "increase_playback_speed"
        elif np.array_equal(fingers_up_array, [0, 0, 0, 0, 1]):
            return "skip_to_next"
        elif np.array_equal(fingers_up_array, [0, 1, 0, 0, 1]):
            return "like_track"
        elif np.array_equal(fingers_up_array, [1, 1, 1, 0, 0]):
            return "shuffle"
        elif np.array_equal(fingers_up_array, [0, 1, 1, 1, 1]):
            return "repeat"
        elif np.array_equal(fingers_up_array, [1, 0, 0, 0, 1]):
            return "quick_search"
    
    elif current_state == "NAVIGATION":
        if np.array_equal(fingers_up_array, [0, 1, 0, 0, 0]):
            return "scroll_up"
        elif np.array_equal(fingers_up_array, [0, 0, 0, 0, 1]):
            return "scroll_down"
            
    return None

def draw_buttons(frame, buttons):
    """Draws buttons on the frame."""
    for button_name, (x1, y1, x2, y2) in buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, button_name, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# --- Main Program Logic ---
def main():
    """
    Main function to run the hand tracking and Spotify control application.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    current_state = "HOME"
    
    cooldown_period = 2.0
    last_action_time = time.time()
    last_gesture = None

    print("Starting Spotify Hand Controller...")
    print("Move your hand into the camera view.")
    print("Make sure the Spotify tab is the active window for controls to work.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        gesture_text = "No Gesture Detected"
        
        main_menu_buttons = {
            "PLAYBACK": (50, 50, 200, 100),
            "NAVIGATION": (50, 120, 200, 170),
            "HELP": (50, 190, 200, 240)
        }
        back_button = (w - 150, 50, w - 50, 100)

        # Universal gesture check
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                fingers_up_array = count_fingers_up(hand_landmarks)
                if np.array_equal(fingers_up_array, [1, 0, 0, 0, 0]):
                    if current_state != "HOME":
                        current_state = "HOME"
                        return_home() # This hotkey is defined in the initial user request
                        time.sleep(0.5)

        if current_state == "HOME":
            draw_buttons(frame, main_menu_buttons)
            cv2.putText(frame, "Pinch a button to select", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    pinching, pinch_x, pinch_y = is_pinching(hand_landmarks, frame)
                    if pinching:
                        for button_name, (x1, y1, x2, y2) in main_menu_buttons.items():
                            if x1 <= pinch_x <= x2 and y1 <= pinch_y <= y2:
                                print(f"Transitioning to {button_name} state.")
                                current_state = button_name
                                time.sleep(0.5)

        elif current_state == "PLAYBACK":
            cv2.rectangle(frame, back_button[:2], back_button[2:], (0, 0, 255), 2)
            cv2.putText(frame, "BACK", (w - 140, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    pinching, pinch_x, pinch_y = is_pinching(hand_landmarks, frame)
                    if pinching:
                        if back_button[0] <= pinch_x <= back_button[2] and back_button[1] <= pinch_y <= back_button[3]:
                            print("Transitioning to HOME state.")
                            current_state = "HOME"
                            time.sleep(0.5)

                    fingers_up_array = count_fingers_up(hand_landmarks)
                    current_time = time.time()
                    gesture = get_gesture(fingers_up_array, current_state)
                    
                    if gesture:
                        gesture_text = f"Gesture: {gesture.replace('_', ' ').title()}"
                        
                        if gesture in ["raise_volume", "lower_volume"]:
                            if gesture == "raise_volume":
                                raise_volume()
                            elif gesture == "lower_volume":
                                lower_volume()
                        elif current_time - last_action_time > cooldown_period:
                            if gesture == "play_pause": play_pause()
                            elif gesture == "seek_forward": seek_forward()
                            elif gesture == "seek_backward": seek_backward()
                            elif gesture == "decrease_playback_speed": decrease_playback_speed()
                            elif gesture == "increase_playback_speed": increase_playback_speed()
                            elif gesture == "skip_to_previous": skip_to_previous()
                            elif gesture == "skip_to_next": skip_to_next()
                            elif gesture == "like_track": like_track()
                            elif gesture == "shuffle": shuffle()
                            elif gesture == "repeat": repeat()
                            elif gesture == "quick_search": quick_search()
                            
                            last_action_time = current_time
                    else:
                        gesture_text = "No Gesture Detected"

        elif current_state == "NAVIGATION":
            cv2.rectangle(frame, back_button[:2], back_button[2:], (0, 0, 255), 2)
            cv2.putText(frame, "BACK", (w - 140, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    pinching, pinch_x, pinch_y = is_pinching(hand_landmarks, frame)
                    if pinching:
                        if back_button[0] <= pinch_x <= back_button[2] and back_button[1] <= pinch_y <= back_button[3]:
                            print("Transitioning to HOME state.")
                            current_state = "HOME"
                            time.sleep(0.5)

                    fingers_up_array = count_fingers_up(hand_landmarks)
                    gesture = get_gesture(fingers_up_array, current_state)

                    if gesture:
                        gesture_text = f"Gesture: {gesture.replace('_', ' ').title()}"
                        if gesture == "scroll_up":
                            scroll_up()
                        elif gesture == "scroll_down":
                            scroll_down()
                    else:
                        gesture_text = "No Gesture Detected"

        elif current_state == "HELP":
            cv2.rectangle(frame, back_button[:2], back_button[2:], (0, 0, 255), 2)
            cv2.putText(frame, "BACK", (w - 140, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            help_text = [
                "Playback Gestures:",
                "Play/Pause: Fist close",
                "Seek Forward: Middle + Ring fingers",
                "Seek Backward: Index finger",
                "Raise Volume: Index & Middle fingers (continuous)",
                "Lower Volume: Ring & Pinky fingers (continuous)",
                "Decrease Speed: Index & Ring fingers",
                "Increase Speed: Middle & Pinky fingers",
                "Skip Previous: Thumb",
                "Skip Next: Pinky",
                "Like: Index & Pinky",
                "Shuffle: Thumb, Index, Middle",
                "Repeat: Index, Middle, Ring, Pinky",
                "Quick Search: Thumb & Pinky",
                "",
                "Navigation Gestures:",
                "Scroll Up: Index finger (continuous)",
                "Scroll Down: Pinky finger (continuous)",
                "",
                "Pinch the BACK button to return to the Main Menu.",
                "Universal Command: Thumbs Up returns to the main menu."
            ]
            
            for i, text in enumerate(help_text):
                cv2.putText(frame, text, (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    pinching, pinch_x, pinch_y = is_pinching(hand_landmarks, frame)
                    if pinching:
                        if back_button[0] <= pinch_x <= back_button[2] and back_button[1] <= pinch_y <= back_button[3]:
                            print("Transitioning to HOME state.")
                            current_state = "HOME"
                            time.sleep(0.5)

        cv2.putText(frame, f"State: {current_state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Spotify Hand Controller', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    print("Wait 3 seconds before starting...")
    time.sleep(3)
    main()
