#  Adaptive Edge Detection using PSO + Machine Learning  
### (Canny + Sobel + Particle Swarm Optimization + Unet + Vision Transformer)

##  Overview  
This project implements an **Adaptive Edge Detection System** combining:

- **Traditional edge detectors** → *Canny* & *Sobel*  
- **PSO (Particle Swarm Optimization)** → optimizes threshold values  
- **Vision Transformer (ViT)** → learns global image features  
- **U-Net Architecture** → enhances spatial feature extraction and improves segmentation-based edge refinement  
- **Fusion Model** → improves accuracy and adaptability over standard methods  
  

The goal is to achieve:  
 -Higher accuracy  
 -Adaptive thresholding  
 -Better edge quality  
 -Robustness for real-time applications  

---

##  Project Structure  
```
adaptive_edge_detection_project/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── backend/
│   ├── app.py                  # Main backend application / API
│   ├── models.py               # Model loading + architecture definitions
│   ├── generate_masks.py       # Mask generation utilities
│   ├── train_model.py          # Training master script
│   ├── train_edge_model.py     # Edge model training
│   ├── train_unet_model.py     # U-Net training script
│   ├── train_vit_model.py      # ViT training script
│   ├── vit_model.py            # Vision Transformer model
│   ├── threshold_predictor.pkl # Trained threshold prediction model
│   ├── images.db               # Database for storing image metadata
│
│   ├── canny_masks/
│   │   └── ...                 # Auto-generated canny mask outputs
│
│   ├── dataset/
│   │   └── ...                 # Training dataset images
│
│   ├── instance/
│   │   └── images.db           # Instance database
│
│   ├── models/
│   │   ├── edge_detection_model.h5
│   │   ├── vit_unet_model.h5
│   │   ├── vit_unet_weights.h5
│   │   └── threshold_predictor.pkl
│
│   ├── src/
│   │   ├── __init__.py
│   │   ├── edge_detection.py       # Main edge detection logic
│   │   ├── edge_accuracy.py        # Accuracy measurement
│   │   ├── feature_extraction.py   # Feature extractor
│   │   ├── ml_predictor.py         # ML prediction logic
│   │   └── pso_optimizer.py        # Particle Swarm Optimization module
│
│   ├── uploads/
│   │   └── ...                 # Uploaded input images from frontend
│
│
├── frontend/
│   ├── index.html              # Main UI webpage
│   ├── script.js               # Frontend logic (image upload, preview)
│   ├── style.css               # UI styling
│   └── view_images.html        # Page to view saved images
│
├── uploads/
│   └── ...                     # Global uploads folder (if used)
│
└── Edge-detection-using-PSO-and-ML/ (optional folder)
    └── ...                     # Extra materials or report files


---

##  Technologies Used  
- Python, TensorFlow, OpenCV  
- Vision Transformer (ViT)  
- Particle Swarm Optimization  
- Canny & Sobel Edge Detection  
- Optional HTML,CSS,Js/React Frontend  

---

##  Installation  
```bash
pip install -r requirements.txt
pip install flask flask_cors flask_sqlalchemy numpy opencv-python tensorflow pillow 
```

---

##  Running the Project  

This project contains **five training pipelines** and **one preprocessing (mask generation) pipeline**.  
Each script trains a different component of your hybrid Adaptive Edge Detection System.

The generated models are automatically saved inside:

```
backend/models/
```

Below are the correct commands, their purpose, and exactly which `.h5` / `.pkl` file each script generates.

---

###  0. (MANDATORY FIRST STEP) — Generate Canny Masks  
This script creates **ground-truth training masks** needed by U-Net and ViT-UNet.

```bash
python backend/generate_masks.py
```
**What happens:**  
- Reads images from `backend/dataset/`  
- Generates Canny masks using thresholds 50–150  
- Saves them into `backend/canny_masks/`  
- Required before training **U-Net**, **Edge Model**, and **ViT-UNet**

**Files generated:**  
- `.png` / `.jpg` masks inside:  
  ```
  backend/canny_masks/
  ```

---

### 1️. Train the U-Net Model (Segmentation-Based Edge Refinement)  
```bash
python backend/train_unet_model.py
```
**What happens:**  
- Loads dataset + canny masks :contentReference[oaicite:0]{index=0}  
- Builds and trains a **U-Net** segmentation model  
- Learns fine-grained edge maps

**Files generated (saved automatically):**  
```
backend/models/edge_detection_model.h5
```

---

### 2️. Train the Vision Transformer + UNet Hybrid Model (ViT-UNet)  
```bash
python backend/train_vit_model.py
```
**What happens:**  
- Loads dataset + canny masks :contentReference[oaicite:1]{index=1}  
- Builds ViT-UNet using `build_vit_segmentation_model()` from `vit_model.py`  
- Trains Transformer + UNet decoder for deep global-spatial edge detection

**Files generated:**  
```
backend/models/vit_unet_model.h5
backend/models/vit_unet_weights.h5
```

---

### 3️. Train the Traditional + ML Edge Detection Model  
```bash
python backend/train_edge_model.py
```
**What happens:**  
- Loads dataset + Canny masks :contentReference[oaicite:2]{index=2}  
- Builds and trains a **pure U-Net edge model** (not ViT-based)  
- Used for fast inference and baseline comparison

**Files generated:**  
```
backend/models/edge_detection_model.h5
```

---

### 4️. Train the Threshold Predictor Model (RandomForest)  
```bash
python backend/train_model.py
```
**What happens:**  
- Extracts 3 features from each training image (mean, std, entropy)  
- Trains a `RandomForestClassifier` to predict thresholds for PSO/Canny  
- Saves threshold prediction model using pickle :contentReference[oaicite:3]{index=3}

**Files generated:**  
```
backend/models/threshold_predictor.pkl
```

---

###  5. Start server 
After training all the modals Run this command to start the backend.
If your backend run succesfully you will get to see on the screen "Adaptive Edge Detection API — Running".

```bash
python backend/app.py
```

###  6. For frontend(In another terminal)
Open a new terminal to run the frontend,

```bash
.venv\Scripts\Activate
```

Move into **frontend** folder

```bash
cd frontend
```

Run the command

```bash
python -m http.server 3000
```
After running this command go on the Crome and type "localhost://3000" in the search bar, you will find the output.

---

###  Summary of All Training Outputs

| Script | Purpose | Output File(s) |
|--------|---------|----------------|
| generate_masks.py | Create canny ground-truth masks | `backend/canny_masks/*.png` |
| train_unet_model.py | Train U-Net model | `edge_detection_model.h5` |
| train_vit_model.py | Train ViT-UNet hybrid | `vit_unet_model.h5`, `vit_unet_weights.h5` |
| train_edge_model.py | Train classical U-Net edge model | `edge_detection_model.h5` |
| train_model.py | Train threshold predictor (RandomForest) | `threshold_predictor.pkl` |
| app.py | To start backend server | - |

---

###  Recommended Training Order

```
1. python backend/generate_masks.py
2. python backend/train_unet_model.py
3. python backend/train_vit_model.py
4. python backend/train_edge_model.py
5. python backend/train_model.py
6. python backend/app.py
```

This ensures all deep models + threshold predictor are properly trained.



---

##  Contributing  
Contributions are welcome!

---

## 👩‍🎓 Author  
**Renuka Balaji Biradar**  
Final Year B.Tech (CSE)  
