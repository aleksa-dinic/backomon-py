import cv2
import numpy as np

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
# ==================================


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
        
        # Calculate bounding box for this cluster
        min_x = np.min(cl[:, 0])
        max_x = np.max(cl[:, 0])
        min_y = np.min(cl[:, 1])
        max_y = np.max(cl[:, 1])
        
        dice.append([len(cl), centroid[0], centroid[1], min_x, max_x, min_y, max_y])

    return dice


def overlay_info(frame, dice, blobs):
    for b in blobs:
        pos = b.pt
        r = b.size / 2
        cv2.circle(frame, (int(pos[0]), int(pos[1])), int(r), (255, 0, 0), 2)

    for d in dice:
        padding = 20
        x1 = int(d[3] - padding)
        y1 = int(d[5] - padding)
        x2 = int(d[4] + padding)
        y2 = int(d[6] + padding)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        textsize = cv2.getTextSize(str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
        cv2.putText(
            frame, str(d[0]),
            (int(d[1] - textsize[0] / 2), int(d[2] + textsize[1] / 2)),
            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2
        )

    return frame


cap = cv2.VideoCapture("http://192.168.0.29:8080/video")

while True:
    ret, frame = cap.read()

    blobs = get_blobs(frame)
    dice = get_dice_from_blobs(blobs)
    out_frame = overlay_info(frame, dice, blobs)

    cv2.imshow("frame", out_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()