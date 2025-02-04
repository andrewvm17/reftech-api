# app.py

import io
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from werkzeug.exceptions import HTTPException

# Import your detector and clustering logic
from logic.detector_v4 import detector_v4
from logic.line_clustering import cluster_lines, best_fit_lines_from_clusters

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust allowed origins as needed

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logger = logging.getLogger(__name__)

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all uncaught exceptions."""
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    logger.error(f"Unhandled Exception: {e}")
    return jsonify({"error": "Internal Server Error"}), 500

@app.route('/get-lines', methods=['POST'])
def get_lines():
    """
    Process an uploaded image, detect all lines, cluster them, and return a best-fit line per cluster.
    
    Expected:
      - An image file uploaded with key 'image'.
      
    Returns:
      - JSON response with a list of best-fit lines. Each line is represented as:
            { "cluster": <cluster_label>, "x1": ..., "y1": ..., "x2": ..., "y2": ..., "slope": ... }
    """
    try:
        # 1. Retrieve the file from the POST request
        file = request.files.get('image')
        if not file:
            logger.warning("No file uploaded in the request.")
            return jsonify({"error": "No file uploaded"}), 400

        # 2. Validate the file type
        if not allowed_file(file.filename):
            logger.warning(f"Unsupported file type: {file.filename}")
            return jsonify({"error": "Unsupported file type"}), 400

        # 3. Convert the file into a format for OpenCV
        in_memory_file = io.BytesIO(file.read())
        pil_img = Image.open(in_memory_file).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        logger.info("Image received and converted for processing.")

        # 4. Detect all lines using detector_v4
        #    Assume detector_v4 returns lines in the format: [x1, y1, x2, y2, slope]
        detected_lines = detector_v4(cv_img)

        if not detected_lines:
            logger.info("No lines detected in the image.")
            return jsonify({"lines": []}), 200

        # 5. (Optional) Ensure that the detected lines are in pure Python types.
        #    This step converts NumPy types (e.g., np.int32, np.float64) into native Python types.
        lines = []
        for line in detected_lines:
            x1, y1, x2, y2, slope = line
            lines.append([
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                float(slope)
            ])

        # 6. Cluster the lines and compute a best-fit line for each cluster.
        clusters = cluster_lines(lines, eps=0.5, min_samples=2)
        best_fit = best_fit_lines_from_clusters(clusters, ignore_noise=True)

        # 7. Convert the best-fit lines into a JSON-serializable list.
        best_fit_serializable = []
        for cluster_label, line in best_fit.items():
            x1, y1, x2, y2, slope = line
            best_fit_serializable.append({
                "cluster": int(cluster_label),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "slope": float(slope) if slope is not None else None
            })

        return jsonify({"lines": best_fit_serializable}), 200

    except Exception as e:
        logger.exception("Error processing the image.")
        return jsonify({"error": "Failed to process the image"}), 500

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Optional: Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
