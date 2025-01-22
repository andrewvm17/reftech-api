# app.py

import io
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from logic.detector_v2 import detect_lines_with_new_algorithm
import logging
from werkzeug.exceptions import HTTPException

# Optional: For API documentation
# from flask_swagger_ui import get_swaggerui_blueprint

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
# It's better to specify the allowed origins in production
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust "origins" as needed

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logger = logging.getLogger(__name__)

# Optional: Swagger UI Setup
# SWAGGER_URL = '/swagger'
# API_URL = '/static/swagger.json'  # Path to your Swagger spec

# swaggerui_blueprint = get_swaggerui_blueprint(
#     SWAGGER_URL,
#     API_URL,
#     config={
#         'app_name': "CheckIt API"
#     }
# )
# app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

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
    Endpoint to process an uploaded image and detect lines.
    
    Expects:
        - An image file uploaded with the key 'image'.
    
    Returns:
        - JSON response containing line coordinates and slope.
    """
    try:
        # 1. Get the file from the POST request
        file = request.files.get('image')
        if not file:
            logger.warning("No file uploaded in the request.")
            return jsonify({"error": "No file uploaded"}), 400

        # 2. Validate file type (optional but recommended)
        if not allowed_file(file.filename):
            logger.warning(f"Unsupported file type: {file.filename}")
            return jsonify({"error": "Unsupported file type"}), 400

        # 3. Convert the file to a NumPy array for OpenCV
        in_memory_file = io.BytesIO(file.read())
        pil_img = Image.open(in_memory_file).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        logger.info("Image received and converted for processing.")

        # 4. Detect lines using the core logic
        line = detect_lines_with_new_algorithm(cv_img)

        if line:
            # Convert the line tuple to a JSON-serializable format
            x1, y1 = map(int, line[0])
            x2, y2 = map(int, line[1])
            slope = float(line[2])

            logger.info(f"Line detected: ({x1}, {y1}) to ({x2}, {y2}) with slope {slope}")

            return jsonify({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "slope": slope
            }), 200
        else:
            logger.info("No lines detected in the image.")
            return jsonify({"line": None}), 200

    except Exception as e:
        logger.exception("Error processing the image.")
        return jsonify({"error": "Failed to process the image"}), 500

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the uploaded file.
    
    Returns:
        bool: True if allowed, False otherwise.
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Optional: Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Use environment variables for configurations
    # Example: FLASK_ENV=production gunicorn app:app
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
