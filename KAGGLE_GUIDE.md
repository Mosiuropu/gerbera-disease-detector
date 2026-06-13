# 🎯 Step-by-Step Kaggle Training Guide

Follow these steps to train your gerbera disease model **completely free** using Kaggle's GPU.

---

## Step 1: Collect Images (Run Locally)

```bash
cd gerbera-disease-detector
pip install -r requirements.txt

# Collect images from iNaturalist + GBIF
python src/scraper.py --source all --limit 300 --output data/raw

# Split into train/val/test
python -c "
from src.preprocess import split_dataset
from pathlib import Path
split_dataset(Path('data/raw'), Path('data/split'))
"
```

## Step 2: Create Kaggle Dataset

1. Go to [kaggle.com](https://www.kaggle.com) and sign in
2. Click **"New"** → **"New Dataset"**
3. Upload the entire `data/split/` folder
4. Name it something like `gerbera-disease-dataset`
5. Click **"Create Dataset"**

## Step 3: Create Kaggle Notebook

1. Click **"New Notebook"** (top right)
2. Click **"Add Data"** → search for your dataset → click **"Add"**
3. **Enable GPU:** Settings (right sidebar) → Accelerator → **GPU P100** or **T4**
4. Copy the code from `notebooks/gerbera_training.py` into the notebook
5. Update the `DATA_DIR` path:

```python
# Change this to match your dataset path
DATA_DIR = Path("/kaggle/input/gerbera-disease-dataset/split")
```

6. **Run All Cells** (Runtime → Run All)

## Step 4: Download Trained Model

After training completes:
1. In the notebook output, you'll see files saved to `outputs/`
2. Click the **"Output"** tab on the right
3. Download `best_model.pth` and `class_names.json`
4. Place them in your local `models/` and `results/` folders

## Step 5: Run the App Locally

```bash
# Copy model files to the right places
# models/best_model.pth  (from Kaggle output)
# results/class_names.json (from Kaggle output)

python app/app.py
# Opens at http://localhost:7860
```

## Step 6: Deploy Free on Hugging Face

1. Create account at [huggingface.co](https://huggingface.co)
2. Create **New Space** → Choose **Gradio** → Free CPU tier
3. Upload your project files
4. Your app is live! 🎉

---

## 📊 Expected Results

| Metric | Small Dataset (~50/class) | Medium Dataset (~200/class) |
|--------|--------------------------|----------------------------|
| Accuracy | 60-75% | 75-85% |
| Macro F1 | 0.50-0.65 | 0.70-0.80 |
| Training Time | 15-30 min | 45-90 min |

> **Tip:** More images = better performance. Aim for 200+ images per class.
