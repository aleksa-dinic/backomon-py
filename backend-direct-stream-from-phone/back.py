from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import threading
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Global variables
camera_source = None
cap = None
lock = threading.Lock()
is_running = False

def process_frame(frame):
    """Process frame for object detection"""
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
            ratio = w / h
            if 0.85 < ratio < 1.15:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(output, "Dice", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return output

def generate_frames():
    """Generator function for video streaming"""
    global cap, is_running
    
    while is_running:
        with lock:
            if cap is None or not cap.isOpened():
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            output = process_frame(frame)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', output)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index-2.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera with specified source"""
    global camera_source, cap, is_running
    
    data = request.json
    source = data.get('source', '0')
    
    # Try to convert to int if it's a number
    try:
        source = int(source)
    except ValueError:
        pass
    
    with lock:
        # Stop existing camera if running
        if cap is not None:
            cap.release()
        
        # Open new camera
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            return jsonify({'success': False, 'message': f'Failed to open camera: {source}'})
        
        camera_source = source
        is_running = True
    
    return jsonify({'success': True, 'message': f'Camera started: {source}'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera"""
    global cap, is_running
    
    with lock:
        is_running = False
        if cap is not None:
            cap.release()
            cap = None
    
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    """Process a single frame from phone camera"""
    try:
        # Get the uploaded image
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        file = request.files['frame']
        
        # Read image from upload
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            return jsonify({'error': 'Failed to encode frame'}), 500
        
        # Return processed image
        return Response(buffer.tobytes(), mimetype='image/jpeg')
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Get current status"""
    with lock:
        return jsonify({
            'is_running': is_running,
            'camera_source': str(camera_source) if camera_source is not None else None
        })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)