"""
Gerbera Disease Detector - Enhanced Web App
=============================================
Gradio-based web app for farmers to upload flower photos and get
disease/pest diagnosis with treatment recommendations.

Features:
- 📷 Camera/smartphone upload for instant diagnosis
- 🔍 AI-powered disease identification with confidence scores
- 💊 Step-by-step treatment recommendations
- 📊 Severity gauge and urgency indicators
- 📅 Seasonal disease calendar
- 🌤️ Weather-based prevention tips
- 📖 Full disease encyclopedia
- 🔄 Disease comparison tool

Deploy free on Hugging Face Spaces or run locally.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datetime
from PIL import Image

import gradio as gr

from disease_db import (
    DISEASE_CLASSES,
    DISEASE_DB,
    format_treatment_advice,
    get_disease_info,
)

# ---------------------------------------------------------------------------
# Try to import the predictor
# ---------------------------------------------------------------------------
try:
    from predict import GerberaPredictor
    MODEL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MODEL_AVAILABLE = False

DEMO_MODE = not MODEL_AVAILABLE


# ---------------------------------------------------------------------------
# Seasonal Disease Calendar
# ---------------------------------------------------------------------------
def get_seasonal_calendar() -> str:
    """Generate seasonal disease risk calendar based on current month."""
    current_month = datetime.datetime.now().month

    seasons = {
        "spring": (3, 5, "Spring (Mar-May)", [
            ("Aphid", "high", "Aphids multiply rapidly in spring"),
            ("Powdery Mildew", "moderate", "Moderate humidity favors mildew"),
            ("Thrips", "moderate", "Thrips become active"),
        ]),
        "summer": (6, 8, "Summer (Jun-Aug)", [
            ("Spider Mite", "high", "Hot, dry weather favors mites"),
            ("Botrytis", "moderate", "High humidity in monsoon"),
            ("Fusarium", "moderate", "Soil-borne fungi thrive"),
            ("Whitefly", "high", "Peak whitefly season"),
        ]),
        "monsoon": (9, 11, "Monsoon (Sep-Nov)", [
            ("Botrytis", "high", "Gray mold thrives in wet conditions"),
            ("Phytophthora", "high", "Waterlogging risk"),
            ("Bacterial Spot", "high", "Rain splash spreads bacteria"),
            ("Alternaria", "moderate", "Humid conditions favor leaf spot"),
        ]),
        "winter": (12, 2, "Winter (Dec-Feb)", [
            ("Viral Mosaic", "moderate", "Vectors still active in greenhouses"),
            ("Fusarium", "low", "Cool temps slow fungal growth"),
            ("Powdery Mildew", "moderate", "Greenhouse humidity issues"),
        ]),
    }

    current_season = None
    for season_name, (start, end, title, diseases) in seasons.items():
        if start <= current_month <= end:
            current_season = (season_name, title, diseases)
            break

    if not current_season:
        current_season = ("spring", seasons["spring"][2], seasons["spring"][3])

    _, title, diseases = current_season

    text = "# Seasonal Disease Calendar\n\n"
    text += f"## Current Season: {title}\n\n"
    text += f"**Month:** {datetime.datetime.now().strftime('%B %Y')}\n\n"

    text += "### Diseases to Watch For:\n\n"
    for disease, risk, tip in diseases:
        risk_badge = {"high": "HIGH RISK", "moderate": "MODERATE", "low": "LOW"}
        text += f"- **{disease}** -- {risk_badge.get(risk, '')}  \n"
        text += f"  _{tip}_\n\n"

    text += "### Seasonal Prevention Checklist:\n\n"
    if current_season[0] == "spring":
        text += "- Inspect plants weekly for aphids\n"
        text += "- Apply preventive neem oil spray\n"
        text += "- Ensure good air circulation\n"
        text += "- Clean greenhouse from winter debris\n"
    elif current_season[0] == "summer":
        text += "- Mist plants to reduce mite risk\n"
        text += "- Install shade nets (50% shade cloth)\n"
        text += "- Monitor soil moisture daily\n"
        text += "- Use yellow sticky traps for whitefly\n"
    elif current_season[0] == "monsoon":
        text += "- Improve drainage around plants\n"
        text += "- Apply copper fungicide preventively\n"
        text += "- Remove dead/dying leaves daily\n"
        text += "- Avoid overhead watering completely\n"
    elif current_season[0] == "winter":
        text += "- Monitor greenhouse humidity\n"
        text += "- Control insect vectors\n"
        text += "- Inspect new plants for virus symptoms\n"
        text += "- Maintain clean growing environment\n"

    return text


# ---------------------------------------------------------------------------
# Weather-Based Tips
# ---------------------------------------------------------------------------
def get_weather_tips() -> str:
    """Provide weather-based disease prevention tips."""
    month = datetime.datetime.now().month

    if month in (6, 7, 8, 9):  # Monsoon
        tips = [
            ("Heavy Rain Expected", "Apply copper fungicide spray NOW to prevent fungal diseases"),
            ("High Humidity", "Improve ventilation, remove lower leaves for air circulation"),
            ("Gray Mold Risk", "HIGH -- Remove all dead flowers and spent petals daily"),
            ("Bacterial Risk", "Avoid all overhead watering, water only at soil level"),
        ]
    elif month in (3, 4, 5):  # Spring
        tips = [
            ("Warming Temperatures", "Aphids emerging -- check leaf undersides weekly"),
            ("Increasing Sunlight", "Gradually acclimate plants to outdoor conditions"),
            ("Pest Activity Rising", "Set up yellow sticky traps now"),
            ("Prevention Window", "Best time for preventive neem oil application"),
        ]
    elif month in (10, 11, 12):  # Autumn
        tips = [
            ("Cooling Down", "Move sensitive plants to shelter before frost"),
            ("Leaf Drop", "Clean fallen leaves to prevent fungal buildup"),
            ("End-of-Season Review", "Document which varieties resist disease best"),
            ("Greenhouse Prep", "Deep clean greenhouse for winter production"),
        ]
    else:  # Winter
        tips = [
            ("Cold Stress", "Cold-stressed plants are more susceptible to disease"),
            ("Heating Systems", "Ensure even heat distribution to prevent condensation"),
            ("Virus Monitoring", "Check for virus symptoms in greenhouse stock"),
            ("Planning Season", "Order certified disease-free planting material"),
        ]

    text = "# Weather-Based Prevention Tips\n\n"
    text += f"**Month:** {datetime.datetime.now().strftime('%B')}\n\n"

    for title, tip in tips:
        text += f"### {title}\n"
        text += f"{tip}\n\n"

    return text


# ---------------------------------------------------------------------------
# Disease Comparison Tool
# ---------------------------------------------------------------------------
def compare_diseases(disease1: str, disease2: str) -> str:
    """Compare two diseases side by side."""
    info1 = get_disease_info(disease1)
    info2 = get_disease_info(disease2)

    text = "# Disease Comparison\n\n"
    text += f"| Feature | {info1['name']} | {info2['name']} |\n"
    text += "|---------|---------|----------|\n"
    text += f"| **Type** | {info1['type']} | {info2['type']} |\n"
    text += f"| **Pathogen** | {info1['pathogen']} | {info2['pathogen']} |\n"
    text += f"| **Severity** | {info1['severity']} | {info2['severity']} |\n"
    text += f"| **Urgency** | {info1['urgency']} | {info2['urgency']} |\n\n"

    text += f"## {info1['name']} Symptoms\n"
    for s in info1['symptoms']:
        text += f"- {s}\n"

    text += f"\n## {info2['name']} Symptoms\n"
    for s in info2['symptoms']:
        text += f"- {s}\n"

    text += f"\n## {info1['name']} Treatment\n"
    for i, t in enumerate(info1['treatments'], 1):
        text += f"{i}. {t}\n"

    text += f"\n## {info2['name']} Treatment\n"
    for i, t in enumerate(info2['treatments'], 1):
        text += f"{i}. {t}\n"

    return text


# ---------------------------------------------------------------------------
# Demo predictor
# ---------------------------------------------------------------------------
def demo_predict(image: Image.Image) -> list:
    """Demo predictions when no model is available."""
    demo_results = [
        {"class": "healthy", "confidence": 0.72},
        {"class": "powdery_mildew", "confidence": 0.15},
        {"class": "botrytis_blight", "confidence": 0.08},
    ]
    results = []
    for r in demo_results:
        info = get_disease_info(r["class"])
        results.append({
            "class": r["class"],
            "confidence": r["confidence"],
            "display_name": info["name"],
            "disease_info": info,
        })
    return results


# ---------------------------------------------------------------------------
# Initialize predictor
# ---------------------------------------------------------------------------
predictor = None
if MODEL_AVAILABLE:
    try:
        model_path = Path(__file__).parent.parent / "models" / "best_model.pth"
        class_names_path = Path(__file__).parent.parent / "results" / "class_names.json"
        if model_path.exists():
            predictor = GerberaPredictor(
                model_path=str(model_path),
                class_names_path=str(class_names_path) if class_names_path.exists() else None,
            )
        else:
            DEMO_MODE = True
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print(f"Warning: Could not load model: {e}")
        DEMO_MODE = True


# ---------------------------------------------------------------------------
# Prediction handler
# ---------------------------------------------------------------------------
def predict_disease(image: Image.Image) -> tuple:
    """Main prediction function. Returns (advice, alt_text, severity_gauge)."""
    if image is None:
        return "Please upload an image first!", "", ""

    if predictor and not DEMO_MODE:
        results = predictor.predict(image, top_k=3)
    else:
        results = demo_predict(image)

    top = results[0]
    info = top["disease_info"]

    # Format main advice
    advice = format_treatment_advice(top["class"], top["confidence"])

    if DEMO_MODE:
        advice = (
            "DEMO MODE -- No trained model found. Showing sample output.\n"
            "Train the model first by running `python src/train.py`\n\n"
            "---\n\n"
        ) + advice

    # Severity gauge
    severity = info.get("severity", "low")
    severity_gauge = (
        f"### Severity: {severity.upper()}\n\n"
        f"**Pathogen:** {info['pathogen']}  \n"
        f"**Type:** {info['type']}  \n"
        f"**Urgency:** {'Act Now' if info.get('urgency') == 'high' else 'Monitor'}"
    )

    # Format alternatives
    alt_text = "**Alternative Possibilities:**\n\n"
    for i, r in enumerate(results[1:], 2):
        ri = r["disease_info"]
        alt_text += (
            f"**#{i} {r['display_name']}** -- "
            f"{r['confidence']:.1%} confidence\n"
            f"   _Type: {ri['type']} | Severity: {ri['severity']}_\n\n"
        )

    return advice, alt_text, severity_gauge


# ---------------------------------------------------------------------------
# Disease encyclopedia
# ---------------------------------------------------------------------------
def get_disease_encyclopedia() -> str:
    """Return formatted disease encyclopedia."""
    text = "# Gerbera Disease Encyclopedia\n\n"
    text += "Complete guide to gerbera diseases, pests, and their management.\n\n"
    text += "---\n\n"

    for cls in DISEASE_CLASSES:
        info = DISEASE_DB[cls]
        if cls == "healthy":
            continue

        severity_badge = {
            "moderate": "Moderate",
            "high": "High",
        }.get(info["severity"], "Low")

        text += f"## {info['icon']} {info['name']}\n\n"
        text += f"- **Type:** {info['type']}\n"
        text += f"- **Pathogen:** {info['pathogen']}\n"
        text += f"- **Severity:** {severity_badge}\n\n"
        text += "**Symptoms:**\n"
        for s in info["symptoms"]:
            text += f"- {s}\n"
        text += "\n**Treatment:**\n"
        for i, t in enumerate(info["treatments"], 1):
            text += f"{i}. {t}\n"
        text += "\n**Prevention:**\n"
        for p in info["prevention"]:
            text += f"- {p}\n"
        text += "\n---\n\n"

    return text


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    """Build the Gradio application."""

    disease_choices = [cls for cls in DISEASE_CLASSES if cls != "healthy"]

    with gr.Blocks(
        title="Gerbera Disease Detector",
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="orange",
        ),
        css="""
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #2d5016 0%, #4a7c23 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .main-header h1 { margin: 0; font-size: 2em; }
        .main-header p { margin: 5px 0 0 0; opacity: 0.9; }
        footer { display: none !important; }
        """,
    ) as app:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>Gerbera Disease Detector</h1>
            <p>AI-powered disease identification for gerbera flower crops</p>
            <p style="font-size: 0.85em; opacity: 0.8;">
                Upload a photo &rarr; Get instant diagnosis &rarr; Know what to do
            </p>
        </div>
        """)

        if DEMO_MODE:
            gr.Markdown(
                "> **Demo Mode** -- No trained model found. Predictions are simulated.\n"
                "> Train your model by running `python src/train.py` or using the Kaggle notebook."
            )

        with gr.Tabs():
            # ================================================================
            # TAB 1: Disease Detection (Main Feature)
            # ================================================================
            with gr.TabItem("Disease Detection", id="detect"):
                gr.Markdown(
                    "**Instructions:** Upload a clear photo of your gerbera plant "
                    "(leaf, flower, or stem). The AI will identify the disease/pest "
                    "and provide treatment recommendations."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="pil",
                            label="Upload Flower Photo (or use camera)",
                            height=400,
                            sources=["upload", "webcam"],
                        )
                        predict_btn = gr.Button(
                            "Analyze Disease",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        severity_output = gr.Markdown(
                            value="Upload an image to see severity assessment.",
                        )
                        gr.Markdown("---")
                        advice_output = gr.Markdown(
                            value="Upload an image and click **Analyze Disease** to get started.",
                        )
                        gr.Markdown("---")
                        alt_output = gr.Markdown(
                            value="",
                        )

                predict_btn.click(
                    fn=predict_disease,
                    inputs=[input_image],
                    outputs=[advice_output, alt_output, severity_output],
                )

            # ================================================================
            # TAB 2: Disease Encyclopedia
            # ================================================================
            with gr.TabItem("Disease Guide", id="guide"):
                gr.Markdown(get_disease_encyclopedia())

            # ================================================================
            # TAB 3: Seasonal Calendar
            # ================================================================
            with gr.TabItem("Seasonal Calendar", id="seasonal"):
                gr.Markdown(get_seasonal_calendar())
                gr.Markdown("---")
                gr.Markdown(get_weather_tips())

            # ================================================================
            # TAB 4: Disease Comparison
            # ================================================================
            with gr.TabItem("Compare Diseases", id="compare"):
                gr.Markdown("### Compare Two Diseases Side by Side")
                with gr.Row():
                    disease1 = gr.Dropdown(
                        choices=disease_choices,
                        label="Disease 1",
                        value="powdery_mildew",
                    )
                    disease2 = gr.Dropdown(
                        choices=disease_choices,
                        label="Disease 2",
                        value="botrytis_blight",
                    )
                compare_btn = gr.Button("Compare", variant="secondary")
                compare_output = gr.Markdown()

                compare_btn.click(
                    fn=compare_diseases,
                    inputs=[disease1, disease2],
                    outputs=[compare_output],
                )

            # ================================================================
            # TAB 5: Quick Treatment Reference
            # ================================================================
            with gr.TabItem("Quick Treatment", id="treatment"):
                gr.Markdown("### Select a disease for quick treatment reference")
                quick_disease = gr.Dropdown(
                    choices=DISEASE_CLASSES,
                    label="Select Disease",
                    value="healthy",
                )
                quick_output = gr.Markdown()

                def show_quick_treatment(cls_name):
                    return format_treatment_advice(cls_name, 1.0)

                quick_disease.change(
                    fn=show_quick_treatment,
                    inputs=[quick_disease],
                    outputs=[quick_output],
                )

            # ================================================================
            # TAB 6: About
            # ================================================================
            with gr.TabItem("About", id="about"):
                gr.Markdown("""
                ## About Gerbera Disease Detector

                This AI-powered tool helps gerbera (and other flower crop) farmers
                quickly identify diseases and pest attacks by simply uploading a photo.

                ### How It Works
                1. **Upload** a clear photo of the affected plant part
                2. **AI analyzes** the image using a deep learning model (EfficientNet-B0)
                3. **Get diagnosis** with disease name, severity, and pathogen info
                4. **Follow treatment** recommendations with step-by-step instructions

                ### Supported Diseases & Pests (12 classes)
                | Class | Type | Severity |
                |-------|------|----------|
                | Healthy | -- | -- |
                | Powdery Mildew | Fungal | Moderate |
                | Botrytis Blight | Fungal | High |
                | Fusarium Rot | Fungal | High |
                | Phytophthora Rot | Oomycete | High |
                | Alternaria Leaf Spot | Fungal | Moderate |
                | Bacterial Leaf Spot | Bacterial | Moderate |
                | Viral Mosaic | Viral | High |
                | Aphid Damage | Pest | Moderate |
                | Whitefly Damage | Pest | Moderate |
                | Thrips Damage | Pest | Moderate |
                | Spider Mite Damage | Pest | Moderate |

                ### Technology
                - **Model:** EfficientNet-B0 (transfer learning from ImageNet)
                - **Training:** Transfer learning with data augmentation
                - **Framework:** PyTorch + Albumentations
                - **App:** Gradio (deployed on Hugging Face Spaces)

                ### Credits
                - Built by [Mosiuropu](https://github.com/Mosiuropu)
                - Part of the Flower Disease Detection Research Project
                - Open source under MIT License
                """)

        return app


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
