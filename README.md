# 🟢 Colored Polygon UNet – Ayna Internship Assignment

## 🔍 Problem Statement
Design and implement a Conditional UNet model that takes:
- A **grayscale polygon image**
- A **color name** (e.g., red, green)

And outputs:
- A **colored polygon image**, where the shape is filled with the specified color.

---

## 🧠 Model Architecture

- Based on the **UNet architecture**
- **Input**: 4 channels — 1 grayscale image + 3 color channels (RGB encoding of color name)
- **Output**: 3-channel RGB image
- Final activation: `Sigmoid`
- Batch Normalization applied after each convolution for stability
- Loss: `L1Loss()` for sharper edge fidelity

---

## ⚙️ Training Details

- **Optimizer**: Adam (`lr = 1e-3`)
- **Loss Function**: `L1Loss()` + **mask-aware focus** (loss computed only on the polygon region)
- **Epochs**: 25
- **Batch Size**: 16
- Tracked with [Weights & Biases](https://wandb.ai/)

Wandb project:  
🔗 [colored-polygon-unet](https://wandb.ai/hari2006006-/colored-polygon-unet)

---

## 📁 Dataset Structure

dataset/
├── training/
│ ├── inputs/ # Grayscale polygon images
│ ├── outputs/ # Ground truth colored polygon images
│ └── data.json # [{"input_polygon": ..., "colour": ..., "output_image": ...}]
├── validation/
│ ├── inputs/
│ ├── outputs/
│ └── data.json

yaml
Copy
Edit

---

## 🧪 Inference Notebook

- File: `inference.ipynb`
- Accepts:
  - A polygon image path
  - A color name
- Outputs the same shape filled with the specified color
- Visualizes both input and output side-by-side

---

## 📈 Training Insights

| Metric     | Trend |
|------------|-------|
| `train_loss` | ↓ steadily over epochs |
| `val_loss`   | ↓ stable and generalizing well |

Loss curves available on wandb dashboard.

---

## 💡 Key Learnings

- Learned how to **condition** UNet using additional semantic input (color)
- Implemented **mask-aware loss** to focus training on meaningful regions
- Explored architectural tweaks like **BatchNorm** and **L1 vs MSE** loss
- Learned to track experiments using **wandb**
- Developed a full workflow from data loading to inference

---

## 🚧 Limitations / Failure Modes

- Model sometimes struggles with **complex overlapping shapes**
- Mild **blurring** at edges due to upsampling
- Background may get faint color bleed in rare cases

---

## ✅ Deliverables

- `dataset.py` – Data loader with mask support
- `unet.py` – UNet model with BatchNorm + Sigmoid
- `train.py` – Mask-aware loss + wandb logging
- `inference.ipynb` – Visual output testing
- `unet_polygon.pth` – Trained model weights
- `README.md` – Insight and workflow summary (this file)

---

© 2025 | Harisiddarth S ✨
