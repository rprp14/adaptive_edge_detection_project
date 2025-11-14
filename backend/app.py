#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os
import io
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from vit_edge import ViTEdgeDetector
from vit_pso import ViTPSO
# NEW: Import the custom fracture predictor
#from fracture_predictor import FracturePredictor 

# ------------------ Suppress TensorFlow INFO/WARNING ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------- App & DB setup ----------------------
app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///images.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------- DB Model ----------------------
class UploadedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<UploadedImage {self.filename}>"

# ---------------------- Edge Detection Methods ----------------------
def basic_edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    out_path = os.path.join(UPLOAD_FOLDER, 'basic_' + os.path.basename(image_path))
    cv2.imwrite(out_path, edges)
    return out_path

def adaptive_edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    out_path = os.path.join(UPLOAD_FOLDER, 'adaptive_' + os.path.basename(image_path))
    cv2.imwrite(out_path, edges)
    return out_path

def sobel_edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    out_path = os.path.join(UPLOAD_FOLDER, 'sobel_' + os.path.basename(image_path))
    cv2.imwrite(out_path, sobel_combined) # type: ignore
    return out_path

def pso_edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150)
    out_path = os.path.join(UPLOAD_FOLDER, 'pso_' + os.path.basename(image_path))
    cv2.imwrite(out_path, edges)
    return out_path

# ---------------------- ML Edge Detection (Optional) ----------------------
model = None
MODEL_PATH = 'edge_detection_model.h5'
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Warning: failed to load ML model: {e}")
        model = None

def ml_edge_detection(image_path):
    if model is None:
        raise RuntimeError("ML model not available")
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (256, 256))
    image_input = np.expand_dims(image_resized/255.0, axis=0)
    edge_map = model.predict(image_input)[0]
    edge_map = np.uint8(255 * (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8))
    edge_map = cv2.dilate(edge_map, np.ones((2,2), np.uint8), iterations=1)
    out_path = os.path.join(UPLOAD_FOLDER, 'ml_' + os.path.basename(image_path))
    cv2.imwrite(out_path, edge_map)
    return out_path

# ---------------------- Initialize ViT and PSO ----------------------
VIT_DEVICE = 'cpu'
vit_detector = ViTEdgeDetector(model_name='vit_base_patch16_224', device=VIT_DEVICE)
vit_pso = ViTPSO(device=VIT_DEVICE, vit_model_name='vit_base_patch16_224')

# Initialize the fracture predictor (loads the H5 file)
#FRACTURE_MODEL_PATH = 'fracture_detection_model.h5'
#fracture_model = FracturePredictor(model_path=FRACTURE_MODEL_PATH)
#fracture_model = FracturePredictor(model_path='fracture_detection_model.h5')

# ---------------------- NEW: Fracture Analysis Wrapper ----------------------
'''def fracture_analysis(image_path):
    """
    Wrapper to use the FracturePredictor class to get status and percentage.
    """
    try:
        return fracture_model.analyze(image_path)
    except Exception as e:
        print(f"Error running fracture analysis: {e}")
        return {"status": "Analysis Failed", "percentage": 0.0}'''

# ---------------------- NEW: ViT Edge Detection Wrappers ----------------------
def vit_edge_detection(image_path):
    """
    Wrapper to use the ViTEdgeDetector class and save the output.
    This calls the run_vit_edge_detection_wrapper in vit_edge.py.
    """
    try:
        base_name = os.path.basename(image_path)
        out_name = 'vit_' + base_name
        out_path = os.path.join(UPLOAD_FOLDER, out_name)
        
        # Calls the function implemented in vit_edge.py
        vit_detector.run_vit_edge_detection_wrapper(image_path, out_path)
        return out_path
    except Exception as e:
        print(f"Error running ViT edge detection: {e}")
        return None

def pso_vit_edge_detection(image_path):
    """
    Wrapper to use the ViTPSO class and save the optimized output.
    """
    try:
        base_name = os.path.basename(image_path)
        out_name = 'pso_vit_' + base_name
        out_path = os.path.join(UPLOAD_FOLDER, out_name)

        pil = Image.open(image_path).convert('RGB')
        gray = np.array(pil.convert('L'))
        # Using Canny as a ground truth target for PSO
        target = cv2.Canny(gray, 100, 200) 

        # Running the PSO optimization
        # NOTE: This uses the conceptual fit method from vit_pso.py
        result = vit_pso.fit(pil, target, pso_iters=10, particles=12)
        best_edge_bin = result['best_edge']['binary']

        # Save the output
        cv2.imwrite(out_path, best_edge_bin)
        return out_path
    except Exception as e:
        print(f"Error running PSO+ViT edge detection: {e}")
        return None
    
# ---------------------- NEW: Fracture Analysis Function ----------------------
'''def fracture_analysis(image_path):
    """Wrapper to call the FracturePredictor and get status/percentage."""
    # Calls the analyze method defined in fracture_predictor.py
    return fracture_model.analyze(image_path)'''


# ---------------------- Routes ----------------------
@app.route('/')
def home():
    html_form = """
    <h2>Adaptive Edge Detection</h2>
    <form method="POST" action="/upload_html" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Upload & Detect Edges</button>
    </form>
    """
    return html_form

@app.route('/upload_html', methods=['POST'])
def upload_image_html():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    new_image = UploadedImage(filename=file.filename, file_path=filepath)
    db.session.add(new_image)
    db.session.commit()

    # --- NEW: Run Fracture Analysis first ---
    #fracture_data = fracture_analysis(filepath)
    #status_color = "red" if fracture_data['status'] == "Fractured" else "green"


    # Run all edge detections
    basic_output = basic_edge_detection(filepath)
    adaptive_output = adaptive_edge_detection(filepath)
    sobel_output = sobel_edge_detection(filepath)
    pso_output = pso_edge_detection(filepath)
    ml_output = ml_edge_detection(filepath) if model else None

    # --- Execute new methods ---
    vit_output = vit_edge_detection(filepath)
    pso_vit_output = pso_vit_edge_detection(filepath)

    # --- Build the HTML output, inserting fracture data immediately after Original image ---
    
    # Determine the color based on status for better visual cue
    #status_color = 'red' if fracture_data['status'] == 'Fractured' else 'green'

    html = f"""
    <h2>Edge Detection Results</h2>
    <p>Original:</p><img src="/get_output/{file.filename}" width="300"/>

    <!-- NEW: Fracture Status and Percentage Display (Inserted here) -->
    <div style="font-family: sans-serif; margin-top: 10px; margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 8px;">
        <p style="font-size: 1.1em; font-weight: bold; margin: 0 0 5px 0;">
             Detected edges using various techniques.
            
        </p>
        <p style="font-size: 1.1em; font-weight: bold; margin: 0;">
             
           
        </p>
    </div>
    <hr style="margin: 20px 0; border: 0; border-top: 1px solid #eee;">
    <!-- END NEW SECTION -->

    <p>Basic Edge:</p><img src="/get_output/{os.path.basename(basic_output)}" width="300"/>
    <p>Adaptive Edge:</p><img src="/get_output/{os.path.basename(adaptive_output)}" width="300"/>
    <p>Sobel Edge:</p><img src="/get_output/{os.path.basename(sobel_output)}" width="300"/>
    <p>PSO Edge:</p><img src="/get_output/{os.path.basename(pso_output)}" width="300"/>
    """
    if ml_output:
        html += f'<p>ML Edge:</p><img src="/get_output/{os.path.basename(ml_output)}" width="300"/>'
    
    # --- Add ViT output to HTML stack ---
    if vit_output:
        html += f'<p>VIT Edge:</p><img src="/get_output/{os.path.basename(vit_output)}" width="300"/>'

    # --- Add PSO+ViT output to HTML stack ---
    if pso_vit_output:
        html += f'<p>PSO+VIT Edge:</p><img src="/get_output/{os.path.basename(pso_vit_output)}" width="300"/>'

    html += '<br><a href="/">Upload another image</a>'
    return html

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error':'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error':'No selected file'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    new_image = UploadedImage(filename=file.filename, file_path=filepath)
    db.session.add(new_image)
    db.session.commit()

    # --- NEW: Run Fracture Analysis first ---
    #fracture_data = fracture_analysis(filepath)

    basic_output = basic_edge_detection(filepath)
    adaptive_output = adaptive_edge_detection(filepath)
    sobel_output = sobel_edge_detection(filepath)
    pso_output = pso_edge_detection(filepath)
    ml_output = ml_edge_detection(filepath) if model else None

    # --- Execute new methods for API response ---
    vit_output = vit_edge_detection(filepath)
    pso_vit_output = pso_vit_edge_detection(filepath)

    resp = {
        'message':'Image uploaded and processed successfully!',
        'image_id': new_image.id,
        'filename': new_image.filename,
        'file_path': new_image.file_path,
        # NEW: Include Fracture Data in API response
        #'fracture_status': fracture_data['status'],
        #'fracture_percentage': fracture_data['percentage'],
        'basic_output': f'{request.host_url}get_output/{os.path.basename(basic_output)}',
        'adaptive_output': f'{request.host_url}get_output/{os.path.basename(adaptive_output)}',
        'sobel_output': f'{request.host_url}get_output/{os.path.basename(sobel_output)}',
        'pso_output': f'{request.host_url}get_output/{os.path.basename(pso_output)}',
        'ml_output': f'{request.host_url}get_output/{os.path.basename(ml_output)}' if ml_output else None,

        # --- Add ViT output to API response ---
        'vit_output': f'{request.host_url}get_output/{os.path.basename(vit_output)}' if vit_output else None,
        # --- Add PSO+ViT output to API response ---
        'pso_vit_output': f'{request.host_url}get_output/{os.path.basename(pso_vit_output)}' if pso_vit_output else None
    }
    return jsonify(resp)

# ---------------------- ViT Edge Routes ----------------------
@app.route('/edge/vit', methods=['POST'])
def edge_vit_route():
    if 'image' not in request.files:
        return jsonify({'error':'No image uploaded'}), 400
    file_content = request.files['image'].read()
    #pil = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
    pil = Image.open(io.BytesIO(file_content)).convert('RGB')
    rollout = int(request.form.get('rollout_start_layer', 0))
    threshold = float(request.form.get('threshold', 0.2))
    smooth_sigma = float(request.form.get('smooth_sigma', 1.0))

    out = vit_detector.get_edge_map(pil, rollout, threshold, smooth_sigma)
    binary = out['binary']

    out_name = f"vit_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.png"
    out_path = os.path.join(UPLOAD_FOLDER, out_name)
    cv2.imwrite(out_path, binary)

    return jsonify({'message':'ViT edge produced',
                    'vit_output': f'{request.host_url}get_output/{out_name}'})

@app.route('/edge/vit_pso', methods=['POST'])
def edge_vit_pso_route():
    if 'image' not in request.files:
        return jsonify({'error':'No image uploaded'}), 400
    file_content = request.files['image'].read()
    #pil = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
    pil = Image.open(io.BytesIO(file_content)).convert('RGB')
    gray = np.array(pil.convert('L'))
    canny_low = int(request.form.get('canny_low', 100))
    canny_high = int(request.form.get('canny_high', 200))
    target = cv2.Canny(gray, canny_low, canny_high)

    pso_iters = int(request.form.get('pso_iters', 10))
    particles = int(request.form.get('particles', 12))
    result = vit_pso.fit(pil, target, pso_iters=pso_iters, particles=particles)
    best_edge_bin = result['best_edge']['binary']

    out_name = f"vit_pso_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.png"
    out_path = os.path.join(UPLOAD_FOLDER, out_name)
    cv2.imwrite(out_path, best_edge_bin)

    return jsonify({'message':'ViT + PSO completed',
                    'vit_pso_output': f'{request.host_url}get_output/{out_name}',
                    'best_params': result.get('best_params'),
                    'best_score': result.get('best_score')})

# ---------------------- Output Endpoints ----------------------
@app.route('/get_output/<filename>')
def get_output(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/download_output/<filename>')
def download_output(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# ---------------------- DB Setup ----------------------
def create_tables():
    with app.app_context():
        db.create_all()

# ---------------------- Run ----------------------
if __name__ == '__main__':
    create_tables()
    app.run(debug=True)



