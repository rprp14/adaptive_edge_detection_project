#  Adaptive Edge Detection using PSO + Machine Learning  
### (Canny + Sobel + Particle Swarm Optimization + Vision Transformer)

##  Overview  
This project implements an **Adaptive Edge Detection System** combining:

- **Traditional edge detectors** → *Canny* & *Sobel*  
- **PSO (Particle Swarm Optimization)** → optimizes threshold values  
- **Vision Transformer (ViT)** → learns global image features  
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

## 🛠️ Technologies Used  
- Python, TensorFlow, OpenCV  
- Vision Transformer (ViT)  
- Particle Swarm Optimization  
- Canny & Sobel Edge Detection  
- Optional HTML/React Frontend  

---

## 📦 Installation  
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project  

### 1️⃣ Train the Model  
```bash
python backend/train_vit_pso_model.py
```

### 2️⃣ Run Edge Detection  
```bash
python backend/pso_optimizer.py
```

### 3️⃣ Optional Frontend  
```bash
python -m http.server 3000
```

---

## 🤝 Contributing  
Contributions are welcome!

---

## 👩‍🎓 Author  
**Renuka Balaji Biradar**  
Final Year B.Tech (CSE)  
