---
title: Gerbera Disease Detector
emoji: 🌻
colorFrom: green
colorTo: orange
sdk: gradio
sdk_version: 6.18.0
python_version: '3.13'
app_file: app/app.py
pinned: false
---

# 🌻 Gerbera Disease Detector

**AI-powered disease identification for gerbera flower crops** — Train a deep learning model to identify diseases and pests in gerbera plants from photos. Farmers upload a picture and get instant diagnosis with treatment recommendations.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Goal

When a farmer uploads a photo of a gerbera crop, the system:
1. **Identifies** what disease or pest attack has occurred
2. **Explains** the severity and symptoms detected
3. **Recommends** specific organic/chemical treatments
4. **Guides** prevention measures for the future

## 🏗️ Architecture

```
Input Image → EfficientNet-B0 (Transfer Learning) → 12-class Prediction → Treatment DB → Farmer Advisory
```

- **Model:** EfficientNet-B0 pretrained on ImageNet, fine-tuned on gerbera disease data
- **Classes:** 12 (1 healthy + 6 fungal/bacterial/viral + 4 pest + 1 oomycete)
- **Training:** Free GPU via Kaggle Notebooks (P100/T4, 30hrs/week)
- **Deployment:** Gradio app on Hugging Face Spaces (free tier)

## 📁 Project Structure

```
gerbera-disease-detector/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── scraper.py               # Collect images from iNaturalist + GBIF (FREE)
│   ├── preprocess.py            # Augmentation & dataset splitting
│   ├── disease_db.py            # Disease database with treatments
│   ├── train.py                 # EfficientNet training pipeline
│   └── predict.py               # Inference engine
├── app/
│   └── app.py                   # Gradio web app for farmers
├── notebooks/
│   └── gerbera_training.py      # Kaggle/Colab notebook (run with free GPU)
├── data/                        # Raw & processed images (gitignored)
├── models/                      # Trained model weights (gitignored)
└── results/                     # Training metrics & plots
```

## 🚀 Quick Start (100% Free)

### Step 1: Collect Images
```bash
# Install dependencies
pip install -r requirements.txt

# Scrape gerbera images from free APIs (iNaturalist + GBIF)
python src/scraper.py --source all --limit 500 --output data/raw
```

### Step 2: Prepare Dataset
```bash
# Split into train/val/test (80/10/10)
python -c "
from src.preprocess import split_dataset
from pathlib import Path
split_dataset(Path('data/raw'), Path('data/split'))
"
```

### Step 3: Train Model
```bash
# Option A: Local training
python src/train.py --data_dir data/split --epochs 30

# Option B: Kaggle (RECOMMENDED - free GPU)
# 1. Upload data/split/ to Kaggle as a dataset
# 2. Create new Kaggle Notebook, add your dataset
# 3. Enable GPU: Settings → Accelerator → GPU P100
# 4. Copy notebooks/gerbera_training.py content into notebook
# 5. Set DATA_DIR = '/kaggle/input/your-dataset-name/'
# 6. Run All
```

### Step 4: Run the App
```bash
python app/app.py
# Opens at http://localhost:7860
```

### Step 5: Deploy Free
```bash
# Option A: Hugging Face Spaces (RECOMMENDED)
# 1. Create account at huggingface.co
# 2. Create new Space → Gradio → Free CPU tier
# 3. Push this repo to the Space

# Option B: Streamlit Cloud
# Push to GitHub → streamlit.io/cloud → Deploy
```

## 🌸 Supported Diseases & Pests (12 Classes)

| # | Class | Type | Severity | Pathogen |
|---|-------|------|----------|----------|
| 1 | ✅ Healthy | — | — | No disease |
| 2 | 🦠 Powdery Mildew | Fungal | Moderate | *Golovinomyces cichoracearum* |
| 3 | 🍄 Botrytis Blight | Fungal | **High** | *Botrytis cinerea* |
| 4 | 🪴 Fusarium Rot | Fungal | **High** | *Fusarium oxysporum* |
| 5 | 💧 Phytophthora Rot | Oomycete | **High** | *Phytophthora cryptogea* |
| 6 | 🍂 Alternaria Leaf Spot | Fungal | Moderate | *Alternaria alternata* |
| 7 | 🦠 Bacterial Leaf Spot | Bacterial | Moderate | *Pseudomonas cichorii* |
| 8 | 🧬 Viral Mosaic | Viral | **High** | CMV / INSV / TSWV |
| 9 | 🐛 Aphid Damage | Pest | Moderate | Aphididae family |
| 10 | 🪰 Whitefly Damage | Pest | Moderate | *Trialeurodes vaporariorum* |
| 11 | 🦗 Thrips Damage | Pest | Moderate | Thripidae family |
| 12 | 🕷️ Spider Mite Damage | Pest | Moderate | *Tetranychus urticae* |

## 💡 Free Training Strategy

| Component | Free Solution | Limits |
|-----------|--------------|--------|
| **Data Collection** | iNaturalist + GBIF APIs | Unlimited (be polite with rate limits) |
| **Data Augmentation** | Albumentations (local) | Unlimited |
| **Training GPU** | Kaggle Notebooks | 30 hrs/week, T4/P100 |
| **Training GPU** | Google Colab Free | ~12 hrs/session, T4 |
| **Model** | EfficientNet-B0 (pretrained) | Transfer learning |
| **Deployment** | Hugging Face Spaces | Free CPU, sleeps after inactivity |
| **Web App** | Gradio | Open source, free tier |

## 📊 Expected Performance

With ~200 images per class (augmented to 500+):
- **Target Accuracy:** 75-85% (small dataset, many classes)
- **Target Macro F1:** 0.70-0.80
- **Inference Time:** <100ms per image on CPU

> ⚡ Performance improves significantly with more data. Aim for 500+ original images per class for production use.

## 🔬 Methodology

1. **Data Collection:** Automated scraping from iNaturalist (research-grade observations) and GBIF (biodiversity records) using their free REST APIs
2. **Auto-classification:** Keyword matching on image metadata/descriptions to assign initial disease labels
3. **Augmentation:** Albumentations pipeline (random crop, flip, color jitter, noise, blur, dropout) to expand small datasets
4. **Transfer Learning:** EfficientNet-B0 pretrained on ImageNet, fine-tuned with custom classifier head
5. **Training:** AdamW optimizer, cosine annealing LR, label smoothing, early stopping
6. **Evaluation:** Macro F1 score (handles class imbalance), confusion matrix, per-class metrics

## 🤝 Contributing

This is a research project. Contributions welcome:
- Add more disease images via the scraper
- Improve classification for underrepresented classes
- Add new flower species (lilium, tuberose, rose)
- Improve the Gradio UI

## 📚 References

- [PlantVillage Dataset](https://plantvillage.psu.edu/) - Reference for plant disease detection
- [iNaturalist API](https://api.inaturalist.org/v1/docs/) - Free image source
- [GBIF API](https://techdocs.gbif.org/) - Biodiversity data
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) - Model architecture
- [Albumentations](https://albumentations.ai/) - Image augmentation

## 👨‍💻 Author

**Mosiuropu** - [GitHub](https://github.com/Mosiuropu)

Part of the Flower Disease Detection Research Project

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
