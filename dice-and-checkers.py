import cv2
import numpy as np

# cap = cv2.VideoCapture("http://192.168.0.29:8080/video")
cap = cv2.VideoCapture("http://192.168.44.133:8080/video")


if not cap.isOpened():
    print("Error: Could not open camera stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    output = frame.copy()

    # --- Preprocess for circle detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_for_circles = cv2.GaussianBlur(gray, (7, 7), 2)
    blur_for_dices = cv2.GaussianBlur(gray, (5, 5), 0)


    edges = cv2.Canny(blur_for_dices, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Circle detection (checkers) ---
    circles = cv2.HoughCircles(
        blur_for_circles,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=10,
        param1=40,
        param2=25,
        minRadius=15,
        maxRadius=25
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.putText(output, "Checker", (x - 20, y - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- Square (dice) detection ---

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 800 or area > 5000:
            continue

        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            # aspect ratio ~ 1 → looks like a square (a dice)
            ratio = w / h
            if 0.85 < ratio < 1.15:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(output, "Dice", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Detected Objects", cv2.resize(output, (50, 50)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
