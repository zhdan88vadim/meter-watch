# counter=0; for file in DIFF_test_digits__less__*.png; do mv "$file" "$counter.png"; ((counter++)); done

from flask import Flask, send_file, jsonify
import os
import threading
import random

app = Flask(__name__)

# Configuration
IMAGE_FOLDER = "../raw_images"
CURRENT_INDEX = 0
INDEX_LOCK = threading.Lock()

# Get all image files matching the pattern
def get_image_list():
    """Return sorted list of image paths matching the pattern"""
    pattern = "*.png"
    import glob
    files = glob.glob(os.path.join(IMAGE_FOLDER, pattern))
    return sorted(files)

# Initialize image list
IMAGE_LIST = get_image_list()
MAX_INDEX = len(IMAGE_LIST) - 1

@app.route('/next_image')
def get_next_image():
    """Return the next image and increment counter"""
    # global CURRENT_INDEX

    CURRENT_INDEX = random.randint(0, len(IMAGE_LIST))
    
    if not IMAGE_LIST:
        return jsonify({"error": "No images found"}), 404
    
    # Thread-safe index update
    with INDEX_LOCK:
        current_img = IMAGE_LIST[CURRENT_INDEX]
        # Update index for next request (circular)
        CURRENT_INDEX = (CURRENT_INDEX + 1) % len(IMAGE_LIST)
    
    return send_file(current_img, mimetype='image/png')

@app.route('/current_index')
def get_current_index():
    """Return current index"""
    return jsonify({"current_index": CURRENT_INDEX, "total_images": len(IMAGE_LIST)})

@app.route('/reset')
def reset_index():
    """Reset the counter to 0"""
    global CURRENT_INDEX
    with INDEX_LOCK:
        CURRENT_INDEX = 0
    return jsonify({"status": "reset", "current_index": 0})

@app.route('/jump/<int:index>')
def jump_to_index(index):
    """Jump to specific image index"""
    global CURRENT_INDEX
    
    if not IMAGE_LIST:
        return jsonify({"error": "No images found"}), 404
    
    if 0 <= index < len(IMAGE_LIST):
        with INDEX_LOCK:
            CURRENT_INDEX = index
        return jsonify({"status": "jumped", "current_index": index})
    else:
        return jsonify({"error": f"Index out of range (0-{len(IMAGE_LIST)-1})"}), 400

@app.route('/')
def index():
    """Simple info page"""
    return f"""
    <h1>Image Server</h1>
    <p>Total images: {len(IMAGE_LIST)}</p>
    <p>Current index: {CURRENT_INDEX}</p>
    <p>Access next image: <a href="/next_image">/next_image</a></p>
    <p>Check status: <a href="/current_index">/current_index</a></p>
    <p>Reset counter: <a href="/reset">/reset</a></p>
    """

if __name__ == '__main__':
    print(f"Total images loaded: {len(IMAGE_LIST)}")
    print(f"Starting server at http://localhost:5050")
    print(f"Access next image at http://localhost:5050/next_image")
    app.run(debug=True, host='0.0.0.0', port=5050)