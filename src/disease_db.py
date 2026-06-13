"""
Gerbera Disease & Pest Database
================================
Comprehensive database of gerbera diseases, pests, symptoms, and treatments.
Used by the prediction app to provide actionable advice to farmers.

Classes (12 total):
1. healthy              - No disease detected
2. powdery_mildew       - Golovinomyces cichoracearum
3. botrytis_blight      - Botrytis cinerea (Gray Mold)
4. fusarium_rot         - Fusarium oxysporum / F. solani
5. phytophthora_rot     - Phytophthora cryptogea
6. alternaria_leaf_spot - Alternaria alternata
7. bacterial_leaf_spot  - Pseudomonas cichorii
8. viral_mosaic         - CMV, INSV, TSWV
9. aphid_damage         - Aphididae family
10. whitefly_damage     - Trialeurodes vaporariorum
11. thrips_damage       - Thripidae family
12. spider_mite_damage  - Tetranychus urticae
"""

DISEASE_CLASSES = [
    "healthy",
    "powdery_mildew",
    "botrytis_blight",
    "fusarium_rot",
    "phytophthora_rot",
    "alternaria_leaf_spot",
    "bacterial_leaf_spot",
    "viral_mosaic",
    "aphid_damage",
    "whitefly_damage",
    "thrips_damage",
    "spider_mite_damage",
]

DISEASE_DB = {
    "healthy": {
        "name": "Healthy Gerbera",
        "type": "None",
        "pathogen": "N/A",
        "severity": "none",
        "symptoms": [
            "Vibrant green leaves",
            "Bright, uniformly colored petals",
            "Strong stem structure",
            "No spots, discoloration, or wilting",
        ],
        "treatments": [
            "Continue regular watering schedule",
            "Maintain balanced fertilization (NPK 5-10-5)",
            "Monitor for early signs of disease weekly",
            "Ensure good air circulation in greenhouse",
        ],
        "prevention": [
            "Water at the base, avoid wetting foliage",
            "Maintain humidity between 60-70%",
            "Remove dead leaves and spent flowers promptly",
            "Use disease-free planting material",
        ],
        "urgency": "low",
        "icon": "✅",
    },
    "powdery_mildew": {
        "name": "Powdery Mildew",
        "type": "Fungal Disease",
        "pathogen": "Golovinomyces cichoracearum",
        "severity": "moderate",
        "symptoms": [
            "White/gray powdery coating on leaves and petals",
            "Leaf curling and distortion",
            "Stunted new growth",
            "Premature leaf drop in severe cases",
        ],
        "treatments": [
            "Spray potassium bicarbonate solution (1 tsp per quart water)",
            "Apply neem oil spray (2 tbsp neem oil + 1 tsp dish soap per gallon)",
            "Use sulfur-based fungicide as organic option",
            "Remove and destroy heavily infected leaves",
        ],
        "prevention": [
            "Improve air circulation around plants",
            "Reduce humidity below 60%",
            "Avoid overhead watering",
            "Space plants adequately for airflow",
        ],
        "urgency": "moderate",
        "icon": "🦠",
    },
    "botrytis_blight": {
        "name": "Botrytis Blight (Gray Mold)",
        "type": "Fungal Disease",
        "pathogen": "Botrytis cinerea",
        "severity": "high",
        "symptoms": [
            "Fuzzy gray-brown mold on petals and leaves",
            "Water-soaked spots that quickly expand",
            "Soft, mushy rotting of flower petals",
            "Small black sclerotia (survival structures) may form",
        ],
        "treatments": [
            "Remove ALL infected plant material immediately",
            "Apply Bacillus subtilis-based bio-fungicide",
            "Use iprodione or boscalid fungicide if available",
            "Reduce humidity and improve ventilation urgently",
        ],
        "prevention": [
            "Avoid overhead irrigation completely",
            "Remove dead flowers and leaves daily",
            "Maintain humidity below 65%",
            "Ensure adequate spacing between plants",
        ],
        "urgency": "high",
        "icon": "🍄",
    },
    "fusarium_rot": {
        "name": "Fusarium Crown/Root Rot",
        "type": "Fungal Disease",
        "pathogen": "Fusarium oxysporum / F. solani",
        "severity": "high",
        "symptoms": [
            "Progressive yellowing of lower leaves",
            "Wilting that doesn't recover with watering",
            "Brown/reddish discoloration at the crown base",
            "Plant collapse in advanced stages",
        ],
        "treatments": [
            "Remove and destroy severely infected plants",
            "Drench soil with Trichoderma-based bio-fungicide",
            "Apply azoxystrobin fungicide drench to soil",
            "Repot in fresh, sterilized growing medium",
        ],
        "prevention": [
            "Use well-drained, sterilized soil mix",
            "Avoid overwatering - let top inch dry between watering",
            "Ensure drainage holes are not blocked",
            "Sterilize pots and tools between plants",
        ],
        "urgency": "high",
        "icon": "🪴",
    },
    "phytophthora_rot": {
        "name": "Phytophthora Root/Crown Rot",
        "type": "Oomycete Disease",
        "pathogen": "Phytophthora cryptogea",
        "severity": "high",
        "symptoms": [
            "Sudden wilting despite adequate soil moisture",
            "Dark, water-soaked lesions at the crown",
            "Root system turns brown and mushy",
            "Stunted growth and eventual plant death",
        ],
        "treatments": [
            "Remove infected plants to prevent spread",
            "Apply metalaxyl or fosetyl-aluminum fungicide",
            "Improve soil drainage with perlite/vermiculite",
            "Use phosphorous acid-based drench as preventive",
        ],
        "prevention": [
            "Never allow plants to sit in standing water",
            "Use raised beds or elevated containers",
            "Sterilize growing media before use",
            "Avoid contaminated irrigation water",
        ],
        "urgency": "high",
        "icon": "💧",
    },
    "alternaria_leaf_spot": {
        "name": "Alternaria Leaf Spot",
        "type": "Fungal Disease",
        "pathogen": "Alternaria alternata",
        "severity": "moderate",
        "symptoms": [
            "Small circular dark brown/black spots on leaves",
            "Concentric ring pattern within spots (target-like)",
            "Yellow halo surrounding each spot",
            "Spots may merge, causing large necrotic areas",
        ],
        "treatments": [
            "Remove infected leaves promptly",
            "Apply copper-based fungicide spray",
            "Use chlorothalonil fungicide for severe cases",
            "Spray baking soda solution (1 tbsp per gallon + soap)",
        ],
        "prevention": [
            "Water at base of plant, avoid leaf wetness",
            "Maintain good air circulation",
            "Mulch to prevent soil splash onto leaves",
            "Clean tools with 10% bleach solution",
        ],
        "urgency": "moderate",
        "icon": "🍂",
    },
    "bacterial_leaf_spot": {
        "name": "Bacterial Leaf Spot",
        "type": "Bacterial Disease",
        "pathogen": "Pseudomonas cichorii",
        "severity": "moderate",
        "symptoms": [
            "Small water-soaked spots on leaves",
            "Spots turn dark brown/black with irregular margins",
            "Yellow halos around infected spots",
            "Spots may become angular, bounded by leaf veins",
        ],
        "treatments": [
            "Remove and destroy all infected foliage",
            "Apply copper hydroxide-based bactericide",
            "Avoid working with wet plants (spreads bacteria)",
            "Improve ventilation to reduce leaf wetness",
        ],
        "prevention": [
            "Use drip irrigation, never overhead sprinklers",
            "Sterilize cutting tools between uses",
            "Avoid handling wet plants",
            "Use certified disease-free transplants",
        ],
        "urgency": "moderate",
        "icon": "🦠",
    },
    "viral_mosaic": {
        "name": "Viral Mosaic Disease",
        "type": "Viral Disease",
        "pathogen": "CMV / INSV / TSWV",
        "severity": "high",
        "symptoms": [
            "Mosaic pattern of light/dark green mottling on leaves",
            "Leaf curling, twisting, or strap-like deformity",
            "Color breaking or streaking on flower petals",
            "Overall stunted and distorted growth",
        ],
        "treatments": [
            "NO chemical cure exists - remove and destroy infected plants",
            "Control insect vectors (thrips, aphids, whiteflies)",
            "Use virus-indexed, certified clean planting stock",
            "Disinfect tools with 10% bleach between plants",
        ],
        "prevention": [
            "Buy virus-free certified plants only",
            "Control aphids and thrips proactively",
            "Remove weed hosts around growing area",
            "Quarantine new plants for 2 weeks before introducing",
        ],
        "urgency": "high",
        "icon": "🧬",
    },
    "aphid_damage": {
        "name": "Aphid Infestation",
        "type": "Pest",
        "pathogen": "Aphididae family (multiple species)",
        "severity": "moderate",
        "symptoms": [
            "Curled, distorted, or yellowed young leaves",
            "Sticky honeydew residue on leaf surfaces",
            "Black sooty mold growing on honeydew",
            "Visible small green/black insects on stems and buds",
        ],
        "treatments": [
            "Spray strong jet of water to dislodge aphids",
            "Apply insecticidal soap spray (2 tbsp per quart water)",
            "Use neem oil spray (follow label directions)",
            "Release ladybugs or lacewings as biological control",
        ],
        "prevention": [
            "Inspect new plants before introducing to growing area",
            "Use yellow sticky traps to monitor populations",
            "Avoid excessive nitrogen fertilization (attracts aphids)",
            "Encourage beneficial insects with companion planting",
        ],
        "urgency": "moderate",
        "icon": "🐛",
    },
    "whitefly_damage": {
        "name": "Whitefly Infestation",
        "type": "Pest",
        "pathogen": "Trialeurodes vaporariorum",
        "severity": "moderate",
        "symptoms": [
            "Clouds of tiny white flying insects when plant is disturbed",
            "Sticky honeydew on upper leaf surfaces",
            "Black sooty mold development",
            "Yellowing and wilting of heavily infested leaves",
        ],
        "treatments": [
            "Hang yellow sticky traps near plants",
            "Apply insecticidal soap to leaf undersides",
            "Use neem oil spray every 7-10 days",
            "Introduce Encarsia formosa parasitic wasp (biocontrol)",
        ],
        "prevention": [
            "Check undersides of leaves regularly",
            "Quarantine new plants for 2 weeks",
            "Maintain good ventilation",
            "Avoid overwatering which promotes whitefly breeding",
        ],
        "urgency": "moderate",
        "icon": "🪰",
    },
    "thrips_damage": {
        "name": "Thrips Damage",
        "type": "Pest",
        "pathogen": "Thripidae family (multiple species)",
        "severity": "moderate",
        "symptoms": [
            "Silver or bronze streaking on leaves and petals",
            "Distorted, malformed flower buds",
            "Buds may fail to open",
            "Tiny dark insects visible with magnification",
        ],
        "treatments": [
            "Hang blue sticky traps (thrips are attracted to blue)",
            "Apply spinosad-based insecticide spray",
            "Use neem oil on leaf undersides and buds",
            "Spray insecticidal soap thoroughly on all plant parts",
        ],
        "prevention": [
            "Use blue sticky traps for early detection",
            "Remove weeds that harbor thrips populations",
            "Maintain clean greenhouse environment",
            "Screen ventilation openings with fine mesh",
        ],
        "urgency": "moderate",
        "icon": "🦗",
    },
    "spider_mite_damage": {
        "name": "Spider Mite Damage",
        "type": "Pest",
        "pathogen": "Tetranychus urticae (Two-spotted spider mite)",
        "severity": "moderate",
        "symptoms": [
            "Fine stippling (tiny yellow/white dots) on leaves",
            "Fine webbing on leaf undersides and between stems",
            "Leaves turn bronze/brown and dry out",
            "Severe infestation causes leaf drop and plant decline",
        ],
        "treatments": [
            "Increase humidity around plants (mites dislike moisture)",
            "Spray undersides of leaves with strong water jet",
            "Apply insecticidal soap or miticide spray",
            "Release predatory mites (Phytoseiulus persimilis)",
        ],
        "prevention": [
            "Keep humidity above 60% - mites thrive in dry conditions",
            "Mist plants regularly during dry weather",
            "Inspect leaf undersides weekly with magnifying glass",
            "Isolate new plants to prevent mite introduction",
        ],
        "urgency": "moderate",
        "icon": "🕷️",
    },
}

# Class index mapping for model training
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DISEASE_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# Display names for UI
DISPLAY_NAMES = {
    cls: DISEASE_DB[cls]["name"] for cls in DISEASE_CLASSES
}


def get_disease_info(class_name: str) -> dict:
    """Get full disease information by class name."""
    return DISEASE_DB.get(class_name, DISEASE_DB["healthy"])


def get_urgency_color(urgency: str) -> str:
    """Get color code for urgency level."""
    colors = {
        "low": "green",
        "moderate": "orange",
        "high": "red",
    }
    return colors.get(urgency, "gray")


def format_treatment_advice(class_name: str, confidence: float) -> str:
    """Format a complete treatment recommendation string."""
    info = get_disease_info(class_name)

    if class_name == "healthy":
        return (
            f"✅ **{info['name']}** (Confidence: {confidence:.1%})\n\n"
            f"Your gerbera plant appears healthy! No disease or pest damage detected.\n\n"
            f"**Continue these good practices:**\n"
            + "\n".join(f"• {t}" for t in info["treatments"])
        )

    severity_emoji = "🔴" if info["severity"] == "high" else "🟠" if info["severity"] == "moderate" else "🟢"

    output = (
        f"{info['icon']} **{info['name']}** (Confidence: {confidence:.1%})\n\n"
        f"**Type:** {info['type']}\n"
        f"**Pathogen/Cause:** {info['pathogen']}\n"
        f"**Severity:** {severity_emoji} {info['severity'].upper()}\n\n"
        f"---\n\n"
        f"**🔍 Symptoms Detected:**\n"
        + "\n".join(f"• {s}" for s in info["symptoms"])
        + f"\n\n---\n\n"
        f"**💊 Recommended Treatment (ACT NOW):**\n"
        + "\n".join(f"{i+1}. {t}" for i, t in enumerate(info["treatments"]))
        + f"\n\n---\n\n"
        f"**🛡️ Prevention for Future:**\n"
        + "\n".join(f"• {p}" for p in info["prevention"])
    )

    return output


if __name__ == "__main__":
    print("=" * 60)
    print("GERBERA DISEASE DATABASE")
    print("=" * 60)
    print(f"\nTotal classes: {len(DISEASE_CLASSES)}")
    print(f"\nClasses:")
    for i, cls in enumerate(DISEASE_CLASSES, 1):
        info = DISEASE_DB[cls]
        print(f"  {i:2d}. {info['icon']} {info['name']} ({cls})")
    print("\n" + format_treatment_advice("powdery_mildew", 0.92))
