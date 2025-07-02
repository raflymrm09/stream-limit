# Object Detection Web App using Streamlit

This project is a simple web-based object detection application using Streamlit and PyTorch. It detects objects in uploaded images using a pretrained Faster R-CNN model.

## 📦 Features
- Detects objects from uploaded images
- Runs in a browser using Streamlit
- Uses pretrained Faster R-CNN ResNet50 from Torchvision

## 🚀 How to Run

### Option 1: Run Locally
1. Clone the repository
```bash
git clone https://github.com/yourusername/object-detection-app.git
cd object-detection-app
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the app
```bash
streamlit run app.py
```

### Option 2: Run in Google Colab + Ngrok
1. Copy your app.py and requirements.txt into Colab
2. Run the following cells in Colab:
```python
!pip install streamlit pyngrok torch torchvision Pillow matplotlib
from pyngrok import ngrok

# Simpan kode app
with open("app.py", "w") as f:
    f.write("""
    # paste your full app.py code here
    """)

# Jalankan app
!streamlit run app.py &>/dev/null &
public_url = ngrok.connect(8501)
print("🌐 Buka aplikasimu di:", public_url)
```

## 📷 Sample Output
![screenshot](sample_images/sample1.jpg)

## 📹 Demo Video
Tonton demo di YouTube:
➡️ https://youtube.com/yourvideo

## 🔗 GitHub Repository
➡️ https://github.com/yourusername/object-detection-app
