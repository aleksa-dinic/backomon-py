import cv2
import numpy as np

# ==== DICE DETECTION SETUP ====
params = cv2.SimpleBlobDetector_Params()
params.filterByInertia
params.minInertiaRatio = 0.6
detector = cv2.SimpleBlobDetector_create(params)

# ==== MANUAL DBSCAN REPLACEMENT ====
def cluster_blobs(points, eps=40):
    clusters = []

    for p in points:
        added = False

        for cluster in clusters:
            # Ako je p blizu BAREM jedne tacke u klasteru – pripada tom klasteru
            for q in cluster:
                if np.linalg.norm(p - q) < eps:
                    cluster.append(p)
                    added = True
                    break

            if added:
                break

        if not added:
            clusters.append([p])  # napravi novi klaster

    return clusters

def get_blobs(frame):
    frame_blurred = cv2.medianBlur(frame, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    blobs = detector.detect(frame_gray)
    return blobs

def get_dice_from_blobs(blobs):
    X = []

    for b in blobs:
        pos = np.array(b.pt)
        X.append(pos)

    if len(X) == 0:
        return []

    X = np.asarray(X)

    # --- CUSTOM CLUSTERING ---
    clusters = cluster_blobs(X, eps=40)

    dice = []

    for cl in clusters:
        cl = np.asarray(cl)
        centroid = np.mean(cl, axis=0)
        dice.append([len(cl), centroid[0], centroid[1]])

    return dice

def overlay_dice_info(frame, dice, blobs):
    for b in blobs:
        pos = b.pt
        r = b.size / 2
        cv2.circle(frame, (int(pos[0]), int(pos[1])), int(r), (255, 0, 0), 2)

    for d in dice:
        textsize = cv2.getTextSize(str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
        cv2.putText(
            frame, str(d[0]),
            (int(d[1] - textsize[0] / 2), int(d[2] + textsize[1] / 2)),
            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2
        )

    return frame

def detect_checkers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=10,
        param1=40,
        param2=25,
        minRadius=15,
        maxRadius=25
    )

    return circles

def overlay_checker_info(frame, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(frame, "Checker", (x - 20, y - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame

cap = cv2.VideoCapture("http://192.168.0.29:8080/video")

if not cap.isOpened():
    print("Error: Could not open camera stream.")
    exit()

frame_count = 0
dice = []
blobs = []
circles = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1

    # Process every 5th frame for performance
    if frame_count % 5 == 0:
        blobs = get_blobs(frame)
        dice = get_dice_from_blobs(blobs)
        circles = detect_checkers(frame)

    out_frame = frame.copy()
    
    checker_positions = []
    if circles is not None:
        circles_rounded = np.uint16(np.around(circles))
        for (x, y, r) in circles_rounded[0, :]:
            checker_positions.append((x, y, r))
    
    for d in dice:
        dice_value, dice_x, dice_y = d
        
        is_checker = False
        if dice_value == 1:  
            for (cx, cy, cr) in checker_positions:
                dist = np.sqrt((dice_x - cx)**2 + (dice_y - cy)**2)
                if dist < cr: 
                    is_checker = True
                    break
        
        if not is_checker:
            textsize = cv2.getTextSize(str(dice_value), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
            cv2.putText(
                out_frame, str(dice_value),
                (int(dice_x - textsize[0] / 2), int(dice_y + textsize[1] / 2)),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2
            )
    
    # Draw blobs (but skip those inside checkers)
    for b in blobs:
        pos = b.pt
        r = b.size / 2
        
        is_inside_checker = False
        for (cx, cy, cr) in checker_positions:
            dist = np.sqrt((pos[0] - cx)**2 + (pos[1] - cy)**2)
            if dist < cr:
                is_inside_checker = True
                break
        
        if not is_inside_checker:
            cv2.circle(out_frame, (int(pos[0]), int(pos[1])), int(r), (255, 0, 0), 2)
    
    out_frame = overlay_checker_info(out_frame, circles)

    cv2.imshow("Detected Objects", cv2.resize(out_frame, (1000, 800)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()