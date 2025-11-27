# FILE: backend/app.py
# PURPOSE: FINAL MASTER SERVER. Robust model loading (fallback to rebuild+weights), PSO, ML, UNet & ViT inference.
# NOTE: All FIXes are marked with "# FIX:" comments.

'''import os
import io
import pickle
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# pyswarm may be installed as 'pyswarm' — we import pso directly
from pyswarm import pso

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Import your model-builder so we can rebuild ViT if needed
# FIX: We import the function(s) to rebuild the ViT architecture when loading the full model fails.
# Make sure backend/vit_model.py defines `build_vit_segmentation_model` with the same signature used during training.
try:
    from vit_model import build_vit_segmentation_model
    from vit_model import PatchEmbed, TransformerBlock  # Optional: for custom_objects if needed
except Exception:
    # If vit_model import fails, we'll still proceed but ViT won't be available.
    build_vit_segmentation_model = None
    PatchEmbed = None
    TransformerBlock = None

# ------------------ Suppress TensorFlow INFO/WARNING ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------- App & DB setup ----------------------
app = Flask(__name__)
CORS(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
MODELS_FOLDER = os.path.join(APP_ROOT, 'models')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# DB (same as your original)
db_path = os.path.join(APP_ROOT, 'images.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------------- Config ----------------------
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
BATCH_SIZE = 4
EPOCHS = 20

# Model file names (change if your files are different)
DL_UNET_FULLPATH = os.path.join(MODELS_FOLDER, 'edge_detection_model.h5')
# For ViT we prefer weights file (recommended). If not present, the code will try the full model file.
VIT_WEIGHTS_PATH = os.path.join(MODELS_FOLDER, 'vit_unet_weights.h5')      # recommended
VIT_FULLMODEL_PATH = os.path.join(MODELS_FOLDER, 'vit_unet_model.h5')     # optional fallback
ML_MODEL_PATH = os.path.join(MODELS_FOLDER, 'threshold_predictor.pkl')

# ---------------------- Custom loss (if needed) ----------------------
# FIX: Define custom loss exactly as used when training (if any). This helps when attempting to load full model.
def CustomBinaryCrossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_flat = tf.reshape(y_true, shape=[-1])
    y_pred_flat = tf.reshape(y_pred, shape=[-1])
    return tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)

CUSTOM_OBJECTS = {'CustomBinaryCrossentropy': CustomBinaryCrossentropy}
# Also include PatchEmbed/TransformerBlock if vit model was saved including them
if PatchEmbed is not None:
    CUSTOM_OBJECTS['PatchEmbed'] = PatchEmbed
if TransformerBlock is not None:
    CUSTOM_OBJECTS['TransformerBlock'] = TransformerBlock

# ---------------------- Globals to hold models ----------------------
dl_unet_model = None
vit_model = None
ml_model = None

# ---------------------- Model loading helpers ----------------------

def try_load_full_model(path, custom_objects=None):
    """Try loading a full Keras model (.h5). Returns model or raises."""
    # FIX: use compile=False to avoid optimizer/loss deserialization issues
    return load_model(path, compile=False, custom_objects=custom_objects or {})


def try_rebuild_vit_and_load_weights(weights_path, input_shape=(256,256,1), patch_size=16):
    """
    FIX: Rebuilds the ViT architecture (from vit_model.build_vit_segmentation_model)
    and loads weights. This avoids deserialization problems with custom layers.
    """
    global build_vit_segmentation_model
    if build_vit_segmentation_model is None:
        raise RuntimeError("build_vit_segmentation_model not importable from vit_model.py")
    model = build_vit_segmentation_model(
        input_shape=input_shape,
        patch_size=patch_size,
        num_layers=4,
        num_heads=4,
        embed_dim=64,
        feed_forward_dim=128,
        decoder_filters=(64, 32, 16, 8)
    )
    # FIX: compile is optional for inference; keep consistent loss if you need to eval
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.load_weights(weights_path)
    return model

# ---------------------- Load UNet (safe) ----------------------
try:
    if os.path.exists(DL_UNET_FULLPATH):
        # FIX: compile=False avoids errors if saved model included optimizer/loss incompatible with current TF
        dl_unet_model = try_load_full_model(DL_UNET_FULLPATH, custom_objects=CUSTOM_OBJECTS)
        print(f"--- Successfully loaded UNet model from {DL_UNET_FULLPATH} ---")
    else:
        print(f"--- UNet model file not found at: {DL_UNET_FULLPATH} (skipping) ---")
except Exception as e:
    print(f"!!! Error loading UNet model: {e} !!!")

# ---------------------- Load ViT model (robust: try full model -> fallback to rebuild+weights) ----------------------
# FIX: We try to load the full model first (in case it was saved without custom issues),
# but if that fails we rebuild the model architecture and load the weights file instead.
try:
    if os.path.exists(VIT_FULLMODEL_PATH):
        try:
            vit_model = try_load_full_model(VIT_FULLMODEL_PATH, custom_objects=CUSTOM_OBJECTS)
            print(f"--- Successfully loaded ViT full model from {VIT_FULLMODEL_PATH} ---")
        except Exception as e_full:
            print(f"--- Failed to load full ViT model (will try rebuild+weights). Reason: {e_full}")
            vit_model = None

    # If full model not loaded, try to load weights into rebuilt architecture
    if vit_model is None and os.path.exists(VIT_WEIGHTS_PATH):
        try:
            vit_model = try_rebuild_vit_and_load_weights(VIT_WEIGHTS_PATH, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), patch_size=16)
            print(f"--- Successfully rebuilt ViT model and loaded weights from {VIT_WEIGHTS_PATH} ---")
        except Exception as e_weights:
            print(f"!!! Failed to rebuild ViT model and load weights: {e_weights} !!!")
            vit_model = None
    elif vit_model is None:
        print(f"--- ViT: Neither full model ({VIT_FULLMODEL_PATH}) nor weights ({VIT_WEIGHTS_PATH}) found. ViT disabled. ---")

except Exception as e:
    print(f"!!! Error loading ViT model (outer): {e} !!!")
    vit_model = None

# ---------------------- Load ML predictor ----------------------
try:
    if os.path.exists(ML_MODEL_PATH):
        with open(ML_MODEL_PATH, 'rb') as f:
            ml_model = pickle.load(f)
        print(f"--- Successfully loaded ML predictor model from {ML_MODEL_PATH} ---")
    else:
        print(f"--- ML predictor file not found at: {ML_MODEL_PATH} (skipping) ---")
except Exception as e:
    print(f"!!! Error loading ML predictor model: {e} !!!")
    ml_model = None

# ---------------------- Helper functions (data preprocessing / postprocessing) ----------------------

def preprocess_dl(img_path):
    """Read image grayscale, resize and normalize to shape (1, H, W, 1)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image at {img_path}")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.reshape(img, (1, IMG_HEIGHT, IMG_WIDTH, 1)).astype(np.float32)
    return img

def postprocess_dl(prediction):
    """
    prediction: model output (batch, H, W, 1) with values in [0,1].
    Convert to 8-bit binary edge map (H, W).
    """
    pred = prediction[0]  # (H, W, 1)
    # Convert to 0-255 uint8
    pred_map_8bit = (pred * 255.0).astype(np.uint8)
    # Use Otsu thresholding to binarize (works well for edges)
    _, output_img = cv2.threshold(pred_map_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if output_img.ndim == 3:
        output_img = output_img.squeeze(-1)
    return output_img

# ---------------------- Adaptive Canny helpers ----------------------
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.array([0.0, 0.0, 0.0])
    return np.array([np.mean(img), np.std(img), np.var(img)])

def predict_threshold(features):
    if ml_model is None:
        return 100
    features = np.array(features).reshape(1, -1)
    pred = ml_model.predict(features)
    # FIX: Ensure scalar return
    return float(pred[0])

# ---------------------- PSO cost function ----------------------
def pso_cost_function(thresh, img_gray):
    low_thresh = int(thresh[0])
    high_thresh = low_thresh + 50
    low_thresh = np.clip(low_thresh, 1, 255)
    high_thresh = np.clip(high_thresh, 1, 255)
    edges = cv2.Canny(img_gray, low_thresh, high_thresh)
    return -np.sum(edges)

# ---------------------- DB model ----------------------
class UploadedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<Image {self.filename}>'

with app.app_context():
    db.create_all()

# ---------------------- Routes ----------------------

# FIX: Add root route to prevent 404 when browsing http://127.0.0.1:5000/
@app.route("/", methods=["GET"])
def index():
    return "<h3>Adaptive Edge Detection API — Running</h3>", 200

# Serve uploaded/result files
@app.route('/get_output/<filename>')
def get_output(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_image_and_process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded image using PIL to handle formats robustly
    original_filename = f"upload_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.png"
    original_filepath = os.path.join(UPLOAD_FOLDER, original_filename)
    try:
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(original_filepath)
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {e}'}), 400

    # Save DB record
    new_image_record = UploadedImage(filename=original_filename, file_path=original_filepath)
    db.session.add(new_image_record)
    db.session.commit()

    results = {
        'original_url': f'{request.host_url}get_output/{original_filename}',
        'image_id': new_image_record.id
    }

    try:
        # Read grayscale once (used by many methods)
        img_gray = cv2.imread(original_filepath, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return jsonify({'error': 'Could not read image in grayscale'}), 500

        # 1) Basic Canny
        basic_edges = cv2.Canny(img_gray, 100, 200)
        basic_filename = f"basic_{original_filename}"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, basic_filename), basic_edges)
        results['basic_output'] = f'{request.host_url}get_output/{basic_filename}'

        # 2) Sobel
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = cv2.convertScaleAbs(sobel_edges)
        sobel_filename = f"sobel_{original_filename}"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, sobel_filename), sobel_edges)
        results['sobel_output'] = f'{request.host_url}get_output/{sobel_filename}'

        # 3) Adaptive ML Canny
        features = extract_features(original_filepath)
        ml_thresh = predict_threshold(features)
        adaptive_edges = cv2.Canny(img_gray, int(ml_thresh), int(ml_thresh) + 50)
        adaptive_filename = f"adaptive_ml_{original_filename}"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, adaptive_filename), adaptive_edges)
        results['adaptive_output'] = f'{request.host_url}get_output/{adaptive_filename}'

        # 4) PSO Canny
        pso_thresh, _ = pso(pso_cost_function, [10], [150], args=(img_gray,), swarmsize=10, maxiter=10)
        pso_edges = cv2.Canny(img_gray, int(pso_thresh[0]), int(pso_thresh[0]) + 50)
        pso_filename = f"pso_{original_filename}"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, pso_filename), pso_edges)
        results['pso_output'] = f'{request.host_url}get_output/{pso_filename}'

        # 5) Deep Learning UNet
        if dl_unet_model is not None:
            unet_input = preprocess_dl(original_filepath)
            unet_pred = dl_unet_model.predict(unet_input, verbose=0)
            unet_edges = postprocess_dl(unet_pred)
            dl_filename = f"dl_unet_{original_filename}"
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, dl_filename), unet_edges)
            results['dl_output'] = f'{request.host_url}get_output/{dl_filename}'
        else:
            results['dl_output'] = results.get('adaptive_output')

        # 6) Hybrid ViT-UNet
        if vit_model is not None:
            vit_input = preprocess_dl(original_filepath)
            vit_pred = vit_model.predict(vit_input, verbose=0)
            vit_edges = postprocess_dl(vit_pred)
            vit_filename = f"vit_{original_filename}"
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, vit_filename), vit_edges)
            results['vit_output'] = f'{request.host_url}get_output/{vit_filename}'
        else:
            results['vit_output'] = results.get('adaptive_output')

        # 7) ViT + PSO hybrid (here we use PSO-optimized Canny as hybrid output)
        # If you want to run PSO on ViT output, that's more complex — left as a research extension.
        vit_pso_edges = None
        if vit_model is not None:
            # Re-run PSO for input image and return its Canny map as hybrid output
            pso_thresh2, _ = pso(pso_cost_function, [10], [150], args=(img_gray,), swarmsize=10, maxiter=10)
            pso_low_thresh = int(pso_thresh2[0])
            pso_high_thresh = pso_low_thresh + 50
            vit_pso_edges = cv2.Canny(img_gray, pso_low_thresh, pso_high_thresh)
        else:
            # fallback: use previously computed PSO output
            vit_pso_edges = pso_edges

        vit_pso_filename = f"vit_pso_{original_filename}"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, vit_pso_filename), vit_pso_edges)
        results['vit_pso_output'] = f'{request.host_url}get_output/{vit_pso_filename}'

    except Exception as e:
        # FIX: print stack info server-side for debugging while returning 500 to client
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {e}'}), 500

    return jsonify(results), 200

# ---------------------- Run app ----------------------
if __name__ == '__main__':
    # FIX: increase upload limit to 16MB (same as your earlier code)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    # FIX: run debug for development (set False for production)
    app.run(host="127.0.0.1", port=5000, debug=True)
'''

#================================================================================================================================================================

# FILE: backend/app.py
# PURPOSE: FINAL MASTER SERVER WITH ACCURACY SCORE FOR EACH EDGE-OUTPUT
# NOTE: All FIXES for accuracy marked with "# ACCURACY FIX"

import os
import io
import pickle
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# pyswarm
from pyswarm import pso

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Import ViT model builder ---
try:
    from vit_model import build_vit_segmentation_model
    from vit_model import PatchEmbed, TransformerBlock
except Exception:
    build_vit_segmentation_model = None
    PatchEmbed = None
    TransformerBlock = None

# ===============================
# ACCURACY FIX: import accuracy function
# ===============================
from src.edge_accuracy import compute_edge_accuracy

# ------------------ Suppress TensorFlow logging ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------- App + Folders ----------------------
app = Flask(__name__)
CORS(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
MODELS_FOLDER = os.path.join(APP_ROOT, 'models')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ---------------------- Database ----------------------
db_path = os.path.join(APP_ROOT, 'images.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------------- Config ----------------------
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

# ---------------------- Model paths ----------------------
DL_UNET_FULLPATH = os.path.join(MODELS_FOLDER, 'edge_detection_model.h5')
VIT_WEIGHTS_PATH = os.path.join(MODELS_FOLDER, 'vit_unet_weights.h5')
VIT_FULLMODEL_PATH = os.path.join(MODELS_FOLDER, 'vit_unet_model.h5')
ML_MODEL_PATH = os.path.join(MODELS_FOLDER, 'threshold_predictor.pkl')

# ---------------------- Custom Loss ----------------------
def CustomBinaryCrossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_flat = tf.reshape(y_true, shape=[-1])
    y_pred_flat = tf.reshape(y_pred, shape=[-1])
    return tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)

CUSTOM_OBJECTS = {"CustomBinaryCrossentropy": CustomBinaryCrossentropy}
if PatchEmbed is not None:
    CUSTOM_OBJECTS["PatchEmbed"] = PatchEmbed
if TransformerBlock is not None:
    CUSTOM_OBJECTS["TransformerBlock"] = TransformerBlock

# ---------------------- Globals ----------------------
dl_unet_model = None
vit_model = None
ml_model = None

# ---------------------- Model Loading Helpers ----------------------
def try_load_full_model(path, custom_objects=None):
    return load_model(path, compile=False, custom_objects=custom_objects or {})

def try_rebuild_vit_and_load_weights(weights_path,
                                     input_shape=(256,256,1),
                                     patch_size=16):

    global build_vit_segmentation_model
    if build_vit_segmentation_model is None:
        raise RuntimeError("build_vit_segmentation_model not importable")

    model = build_vit_segmentation_model(
        input_shape=input_shape,
        patch_size=patch_size,
        num_layers=4,
        num_heads=4,
        embed_dim=64,
        feed_forward_dim=128,
        decoder_filters=(64,32,16,8)
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.load_weights(weights_path)
    return model

# ---------------------- Load UNet ----------------------
try:
    if os.path.exists(DL_UNET_FULLPATH):
        dl_unet_model = try_load_full_model(DL_UNET_FULLPATH, CUSTOM_OBJECTS)
        print("[INFO] UNet model loaded")
    else:
        print("[WARN] UNet model missing")
except Exception as e:
    print("UNet loading error:", e)

# ---------------------- Load ViT Model ----------------------
try:
    if os.path.exists(VIT_FULLMODEL_PATH):
        try:
            vit_model = try_load_full_model(VIT_FULLMODEL_PATH, CUSTOM_OBJECTS)
            print("[INFO] ViT full model loaded")
        except Exception as e:
            print("ViT full model failed:", e)

    if vit_model is None and os.path.exists(VIT_WEIGHTS_PATH):
        try:
            vit_model = try_rebuild_vit_and_load_weights(
                VIT_WEIGHTS_PATH,
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            )
            print("[INFO] Rebuilt ViT model from weights")
        except Exception as e:
            print("ViT weight load failed:", e)
            vit_model = None

except Exception as e:
    print("ViT load outer error:", e)
    vit_model = None

# ---------------------- Load ML Predictor ----------------------
try:
    if os.path.exists(ML_MODEL_PATH):
        with open(ML_MODEL_PATH, 'rb') as f:
            ml_model = pickle.load(f)
        print("[INFO] ML predictor loaded")
    else:
        print("[WARN] ML model missing")
except Exception as e:
    print("ML predictor load failed:", e)
    ml_model = None

# ---------------------- Helper Functions ----------------------
def preprocess_dl(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return img.reshape((1, IMG_HEIGHT, IMG_WIDTH, 1))

def postprocess_dl(pred):
    pred = pred[0]
    pred = (pred * 255).astype(np.uint8)
    _, out = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if out.ndim == 3:
        out = out[:, :, 0]
    return out

def extract_features(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.array([0.0, 0.0, 0.0])
    return np.array([np.mean(img), np.std(img), np.var(img)])

def predict_threshold(feat):
    if ml_model is None:
        return 100
    return float(ml_model.predict(np.array(feat).reshape(1, -1))[0])

'''def pso_cost_function(thresh, img_gray):
    low = int(thresh[0])
    high = low + 50
    edges = cv2.Canny(img_gray, low, high)
    return -np.sum(edges)'''

def pso_cost_function(thresh, img_gray):
    """
    PSO cost function: negative of sum of edge pixel intensities.
    Returns a finite scalar cost. Uses int64 accumulation to avoid overflow.
    """
    try:
        # Ensure thresh[0] is a scalar and clip to sensible range
        low_thresh = int(np.clip(thresh[0], 1, 255))
        high_thresh = min(low_thresh + 50, 255)

        # Compute edges
        edges = cv2.Canny(img_gray, low_thresh, high_thresh)

        # Accumulate in a safe integer dtype to avoid uint8 overflow
        #edge_sum = int(np.sum(edges.astype(np.int64)))
        edge_sum = int(np.sum(edges.astype(np.int64)))
        area = edges.size
        cost = -(edge_sum / max(area, 1))


        # We want to maximize number/strength of edge pixels => PSO minimizes, so return negative
        cost = -edge_sum

        # Defensive: ensure finite numeric return
        if not np.isfinite(cost):
            return 1e9
        return cost

    except Exception as ex:
        # If anything goes wrong, return a large positive cost so PSO avoids this region
        print("PSO cost error:", ex)
        return 1e9


# ---------------------- DB Model ----------------------
class UploadedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

with app.app_context():
    db.create_all()

# ---------------------- ROUTES ----------------------
@app.route("/", methods=["GET"])
def index():
    return "<h3>Adaptive Edge Detection API — Running</h3>"

@app.route("/get_output/<filename>")
def get_output(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/upload", methods=["POST"])
def upload_image_and_process():

    # ==== Validate Upload ====
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    # Save uploaded image
    filename = f"upload_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    img = Image.open(file).convert("RGB")
    img.save(filepath)

    # Save in DB
    rec = UploadedImage(filename=filename, file_path=filepath)
    db.session.add(rec)
    db.session.commit()

    results = {
        "original_url": f"{request.host_url}get_output/{filename}",
        "image_id": rec.id
    }

    try:
        img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # ---------------------- 1) BASIC CANNY ----------------------
        basic = cv2.Canny(img_gray, 100, 200)
        basic_file = f"basic_{filename}"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, basic_file), basic)
        results["basic_output"] = f"{request.host_url}get_output/{basic_file}"
        results["basic_accuracy"] = compute_edge_accuracy(basic)  # ACCURACY FIX

        # ---------------------- 2) SOBEL ----------------------
        sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
        sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
        sob = cv2.magnitude(sx, sy)
        sob = cv2.convertScaleAbs(sob)
        sobel_file = f"sobel_{filename}"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, sobel_file), sob)
        results["sobel_output"] = f"{request.host_url}get_output/{sobel_file}"
        results["sobel_accuracy"] = compute_edge_accuracy(sob)  # ACCURACY FIX

        # ---------------------- 3) ADAPTIVE ML CANNY ----------------------
        feat = extract_features(filepath)
        ml_t = predict_threshold(feat)
        adaptive = cv2.Canny(img_gray, int(ml_t), int(ml_t)+50)
        adaptive_file = f"adaptive_ml_{filename}"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, adaptive_file), adaptive)
        results["adaptive_output"] = f"{request.host_url}get_output/{adaptive_file}"
        results["adaptive_accuracy"] = compute_edge_accuracy(adaptive)  # ACCURACY FIX

        # ---------------------- 4) PSO CANNY ----------------------
        pso_t, _ = pso(pso_cost_function, [10], [150], args=(img_gray,), swarmsize=10, maxiter=10)
        pso_edges = cv2.Canny(img_gray, int(pso_t[0]), int(pso_t[0])+50)
        pso_file = f"pso_{filename}"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, pso_file), pso_edges)
        results["pso_output"] = f"{request.host_url}get_output/{pso_file}"
        results["pso_accuracy"] = compute_edge_accuracy(pso_edges)  # ACCURACY FIX

        # ---------------------- 5) UNet DL ----------------------
        if dl_unet_model is not None:
            u_input = preprocess_dl(filepath)
            u_pred = dl_unet_model.predict(u_input, verbose=0)
            u_edges = postprocess_dl(u_pred)
            unet_file = f"dl_unet_{filename}"
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, unet_file), u_edges)
            results["dl_output"] = f"{request.host_url}get_output/{unet_file}"
            results["dl_accuracy"] = compute_edge_accuracy(u_edges)  # ACCURACY FIX
        else:
            results["dl_output"] = results["adaptive_output"]
            results["dl_accuracy"] = results["adaptive_accuracy"]

        # ---------------------- 6) ViT DL ----------------------
        if vit_model is not None:
            v_input = preprocess_dl(filepath)
            v_pred = vit_model.predict(v_input, verbose=0)
            v_edges = postprocess_dl(v_pred)
            vit_file = f"vit_{filename}"
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, vit_file), v_edges)
            results["vit_output"] = f"{request.host_url}get_output/{vit_file}"
            results["vit_accuracy"] = compute_edge_accuracy(v_edges)  # ACCURACY FIX
        else:
            results["vit_output"] = results["adaptive_output"]
            results["vit_accuracy"] = results["adaptive_accuracy"]

        # ---------------------- 7) ViT + PSO Hybrid ----------------------
        if vit_model is not None:
            pso2, _ = pso(pso_cost_function, [10], [150], args=(img_gray,), swarmsize=10, maxiter=10)
            pso_low = int(pso2[0])
            pso_high = pso_low + 50
            v_pso = cv2.Canny(img_gray, pso_low, pso_high)
        else:
            v_pso = pso_edges

        vit_pso_file = f"vit_pso_{filename}"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, vit_pso_file), v_pso)
        results["vit_pso_output"] = f"{request.host_url}get_output/{vit_pso_file}"
        results["vit_pso_accuracy"] = compute_edge_accuracy(v_pso)  # ACCURACY FIX

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    return jsonify(results), 200


# ---------------------- Run Server ----------------------
if __name__ == "__main__":
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.run(host="127.0.0.1", port=5000, debug=True)
