
import cv2
import mediapipe
import pyautogui

# Sensitivity factor for mouse movement
sensitivity = 2.0  # Increase for more sensitivity
sensitivity_step = 0.2  # Amount to change per key press

face_mesh_landmarks = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
prev_mouse_x, prev_mouse_y = None, None

# Adjustable click threshold
click_threshold = 0.01  # Lower for more sensitive click, higher for less

def handle_window_buttons(key):
    # Minimize: m, Maximize: x, Close: c
    if key == ord('m'):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_VISIBLE, 0)
    elif key == ord('x'):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key == ord('c'):
        return True
    return False

window_name = "The eye said, 'I'll be your Mouse'"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, image = cam.read()
    if not ret:
        break
    image = cv2.flip(image, 1)
    window_w, window_h, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmarks.process(rgb_image)
    all_face_landmarks_points = processed_image.multi_face_landmarks
    if all_face_landmarks_points:
        one_face_landmarks_points = all_face_landmarks_points[0].landmark
        for id, landmark_point in enumerate(one_face_landmarks_points[474:478]):
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)
            if id == 1:
                mouse_x = int(screen_w / window_w * x)
                mouse_y = int(screen_h / window_h * y)
                if prev_mouse_x is not None and prev_mouse_y is not None:
                    dx = (mouse_x - prev_mouse_x) * sensitivity
                    dy = (mouse_y - prev_mouse_y) * sensitivity
                    pyautogui.moveRel(dx, dy)
                else:
                    pyautogui.moveTo(mouse_x, mouse_y)
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
            cv2.circle(image, (x, y), 2, (0, 0, 255))
        left_eye = [one_face_landmarks_points[145], one_face_landmarks_points[159]]
        for landmark_point in left_eye:
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)
            cv2.circle(image, (x, y), 2, (0, 255, 255))
        # Improved click function with adjustable threshold
        eye_distance = abs(left_eye[0].y - left_eye[1].y)
        if eye_distance < click_threshold:
            pyautogui.click()
            pyautogui.sleep(0.2)
            print(f"Mouse clicked (eye distance: {eye_distance:.4f})")
    # Display current sensitivity and click threshold on the window
    cv2.putText(image, f'Sensitivity: {sensitivity:.1f} (+/-)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(image, f'Click Threshold: {click_threshold:.3f} ([/])', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(image, 'Minimize[m] Maximize[x] Close[c]', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or handle_window_buttons(key):
        break
    elif key == ord('+') or key == ord('='):
        sensitivity += sensitivity_step
    elif key == ord('-'):
        sensitivity = max(0.1, sensitivity - sensitivity_step)
    elif key == ord('['):
        click_threshold = max(0.001, click_threshold - 0.001)
    elif key == ord(']'):
        click_threshold += 0.001
cam.release()
cv2.destroyAllWindows()