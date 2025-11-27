# 🧠 Adaptive Edge Detection using PSO + Machine Learning  
### (Canny + Sobel + Particle Swarm Optimization + Vision Transformer)

## 📌 Overview  
This project implements an **Adaptive Edge Detection System** combining:

- **Traditional edge detectors** → *Canny* & *Sobel*  
- **PSO (Particle Swarm Optimization)** → optimizes threshold values  
- **Vision Transformer (ViT)** → learns global image features  
- **Fusion Model** → improves accuracy and adaptability over standard methods  

The goal is to achieve:  
✔ Higher accuracy  
✔ Adaptive thresholding  
✔ Better edge quality  
✔ Robustness for real-time applications  

---

## 📁 Project Structure  
```
adaptive_edge_detection_project/
│
├── backend/
│   ├── canny_detector.py
│   ├── sobel_detector.py
│   ├── pso_optimizer.py
│   ├── vit_model.py
│   ├── train_vit_pso_model.py
│   ├── utils.py
│   ├── __init__.py
│   └── sample_images/
│
├── frontend/
│   └── index.html
│
├── requirements.txt
├── README.md
└── run_server.py
```

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
