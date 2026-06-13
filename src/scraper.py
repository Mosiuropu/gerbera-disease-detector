"""
Gerbera Disease Image Scraper
==============================
Collects gerbera images from free public APIs:
1. iNaturalist API (crowdsourced observations)
2. GBIF API (global biodiversity records)

Usage:
    python src/scraper.py --source inaturalist --output data/raw --limit 200
    python src/scraper.py --source gbif --output data/raw --limit 200
    python src/scraper.py --source all --output data/raw --limit 500
"""

import os
import re
import time
import json
import hashlib
import argparse
import requests
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Disease keywords to match against image descriptions/metadata
# These are used to auto-classify downloaded images by disease type
# ---------------------------------------------------------------------------
DISEASE_KEYWORDS = {
    "powdery_mildew": [
        "powdery mildew", "white powder", "gray powder", "fungal coating",
        "white fungus", "gray mold on leaves", "powdery white",
    ],
    "botrytis_blight": [
        "botrytis", "gray mold", "gray mould", "fuzzy gray", "soft rot",
        "brown rot", "petal blight", "flower rot",
    ],
    "fusarium_rot": [
        "fusarium", "crown rot", "root rot", "wilt", "vascular wilt",
        "base rot", "brown crown",
    ],
    "phytophthora_rot": [
        "phytophthora", "root rot", "water-soaked", "sudden wilt",
        "crown rot", "damping off",
    ],
    "alternaria_leaf_spot": [
        "alternaria", "leaf spot", "brown spot", "black spot", "target spot",
        "concentric rings", "circular spot", "necrotic spot",
    ],
    "bacterial_leaf_spot": [
        "bacterial spot", "bacterial leaf", "water-soaked spot",
        "bacterial blight", "angular spot",
    ],
    "viral_mosaic": [
        "mosaic", "virus", "mottling", "yellowing pattern", "streaking",
        "color breaking", "leaf curl", "deformed", "mottled",
    ],
    "aphid_damage": [
        "aphid", "aphids", "green fly", "black fly", "honeydew",
        "sooty mold", "curled leaves", "aphid infestation",
    ],
    "whitefly_damage": [
        "whitefly", "white fly", "whiteflies", "tiny white insect",
        "greenhouse whitefly",
    ],
    "thrips_damage": [
        "thrips", "thrip", "silvering", "bronzing", "flower bud damage",
        "distorted petals",
    ],
    "spider_mite_damage": [
        "spider mite", "mites", "webbing", "stippling", "bronzing",
        "fine webbing", "leaf yellowing", "spotted spider",
    ],
    "healthy": [
        "healthy gerbera", "blooming gerbera", "beautiful gerbera",
        "gerbera daisy", "healthy flower", "vibrant",
    ],
}


def classify_image(description: str) -> Optional[str]:
    """Classify an image based on its description text using keyword matching."""
    if not description:
        return None

    desc_lower = description.lower()
    scores = {}

    for disease, keywords in DISEASE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in desc_lower)
        if score > 0:
            scores[disease] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


def download_image(url: str, save_path: Path, timeout: int = 30) -> bool:
    """Download an image from URL and save it."""
    try:
        headers = {
            "User-Agent": "GerberaDiseaseDetector/1.0 (research; github.com/Mosiuropu)"
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return False

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        # Verify file is not empty and is a valid image (>1KB)
        if save_path.stat().st_size < 1024:
            save_path.unlink()
            return False

        return True
    except Exception as e:
        print(f"  [WARN] Failed to download {url}: {e}")
        return False


# ============================================================================
# iNaturalist Scraper
# ============================================================================
def scrape_inaturalist(
    output_dir: Path,
    limit: int = 500,
    taxon_name: str = "Gerbera",
):
    """
    Download gerbera images from iNaturalist.
    Filters by research-grade observations and attempts to classify by disease keywords.
    """
    print("\n" + "=" * 60)
    print("iNATURALIST IMAGE SCRAPER")
    print("=" * 60)

    api_url = "https://api.inaturalist.org/v1/observations"
    headers = {"User-Agent": "GerberaDiseaseDetector/1.0 (github.com/Mosiuropu)"}

    # First, get the taxon ID for Gerbera
    taxon_url = "https://api.inaturalist.org/v1/taxa"
    taxon_resp = requests.get(
        taxon_url,
        params={"q": taxon_name, "rank": "genus"},
        headers=headers,
        timeout=30,
    )
    taxon_data = taxon_resp.json()

    taxon_id = None
    for result in taxon_data.get("results", []):
        if result.get("name", "").lower() == taxon_name.lower():
            taxon_id = result["id"]
            break

    if not taxon_id and taxon_data.get("results"):
        taxon_id = taxon_data["results"][0]["id"]

    if not taxon_id:
        print("[ERROR] Could not find taxon ID for Gerbera")
        return

    print(f"Found taxon ID for {taxon_name}: {taxon_id}")

    # Create disease subdirectories
    for disease in DISEASE_KEYWORDS:
        (output_dir / disease).mkdir(parents=True, exist_ok=True)

    downloaded = 0
    page = 1
    seen_hashes = set()

    while downloaded < limit:
        print(f"\nFetching page {page} (downloaded: {downloaded}/{limit})...")

        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 100,
            "page": page,
            "order": "desc",
            "order_by": "created_at",
        }

        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] API request failed: {e}")
            break

        results = data.get("results", [])
        if not results:
            print("No more results found.")
            break

        for obs in results:
            if downloaded >= limit:
                break

            # Get description text for classification
            description = obs.get("description", "") or ""
            obs_name = obs.get("taxon", {}).get("name", "")
            common_name = obs.get("taxon", {}).get("preferred_common_name", "")
            tags = " ".join(obs.get("tags", []))
            text = f"{description} {obs_name} {common_name} {tags}"

            # Get observation tags/labels
            observation_fields = obs.get("observation_fields_values", [])

            # Classify based on keywords
            disease_class = classify_image(text)

            # Also check observation field values
            if disease_class is None:
                # Check if any observation fields mention disease
                for ofv in observation_fields:
                    val = str(ofv.get("value", ""))
                    disease_class = classify_image(val)
                    if disease_class:
                        break

            # If still no match, put in healthy if it's a gerbera photo
            if disease_class is None:
                disease_class = "healthy"

            # Download photos
            photos = obs.get("photos", [])
            for photo in photos[:1]:  # Take first photo only
                if downloaded >= limit:
                    break

                img_url = photo.get("url", "")
                if not img_url:
                    continue

                # Request large version
                img_url = img_url.replace("square", "large").replace("small", "large")

                # Generate unique filename
                photo_id = photo.get("id", "")
                img_hash = hashlib.md5(img_url.encode()).hexdigest()[:10]
                ext = ".jpg"
                filename = f"inat_{photo_id}_{img_hash}{ext}"
                save_path = output_dir / disease_class / filename

                if save_path.exists():
                    continue

                # Check for duplicates by content hash
                if img_hash in seen_hashes:
                    continue

                success = download_image(img_url, save_path)
                if success:
                    downloaded += 1
                    seen_hashes.add(img_hash)
                    print(f"  [{downloaded}] {disease_class}/{filename}")

                # Respect rate limits
                time.sleep(0.3)

        page += 1
        time.sleep(1)  # Be polite between pages

    print(f"\n✅ iNaturalist: Downloaded {downloaded} images")


# ============================================================================
# GBIF Scraper
# ============================================================================
def scrape_gbif(
    output_dir: Path,
    limit: int = 500,
    scientific_name: str = "Gerbera",
):
    """
    Download gerbera images from GBIF.
    """
    print("\n" + "=" * 60)
    print("GBIF IMAGE SCRAPER")
    print("=" * 60)

    api_url = "https://api.gbif.org/v1/occurrence/search"

    # Create disease subdirectories
    for disease in DISEASE_KEYWORDS:
        (output_dir / disease).mkdir(parents=True, exist_ok=True)

    downloaded = 0
    offset = 0
    page_size = 300
    seen_hashes = set()

    while downloaded < limit:
        print(f"\nFetching offset {offset} (downloaded: {downloaded}/{limit})...")

        params = {
            "scientificName": scientific_name,
            "mediaType": "stillImage",
            "limit": page_size,
            "offset": offset,
        }

        try:
            resp = requests.get(api_url, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] GBIF API request failed: {e}")
            break

        results = data.get("results", [])
        if not results:
            print("No more results found.")
            break

        for record in results:
            if downloaded >= limit:
                break

            # Get description/label for classification
            label = record.get("label", "") or ""
            basis = record.get("basisOfRecord", "")
            recorded_by = record.get("recordedBy", "") or ""
            event_remarks = record.get("eventRemarks", "") or ""
            identification_notes = record.get("identificationRemarks", "") or ""
            text = f"{label} {basis} {recorded_by} {event_remarks} {identification_notes}"

            # Get media URLs
            media_list = record.get("media", [])
            for media in media_list:
                if downloaded >= limit:
                    break

                if media.get("type") != "StillImage":
                    continue

                img_url = media.get("identifier", "")
                if not img_url:
                    continue

                # Classify
                media_desc = media.get("description", "") or ""
                media_title = media.get("title", "") or ""
                all_text = f"{text} {media_desc} {media_title}"
                disease_class = classify_image(all_text) or "healthy"

                # Download
                img_hash = hashlib.md5(img_url.encode()).hexdigest()[:10]
                if img_hash in seen_hashes:
                    continue

                gbif_id = record.get("key", "unknown")
                filename = f"gbif_{gbif_id}_{img_hash}.jpg"
                save_path = output_dir / disease_class / filename

                if save_path.exists():
                    continue

                success = download_image(img_url, save_path)
                if success:
                    downloaded += 1
                    seen_hashes.add(img_hash)
                    print(f"  [{downloaded}] {disease_class}/{filename}")

                time.sleep(0.2)

        offset += page_size
        time.sleep(1)

    print(f"\n✅ GBIF: Downloaded {downloaded} images")


# ============================================================================
# Summary
# ============================================================================
def print_dataset_summary(data_dir: Path):
    """Print a summary of collected images per class."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    total = 0
    class_counts = {}
    for disease in sorted(DISEASE_KEYWORDS.keys()):
        class_dir = data_dir / disease
        if class_dir.exists():
            count = len(list(class_dir.glob("*.*")))
        else:
            count = 0
        class_counts[disease] = count
        total += count
        bar = "█" * (count // 5) if count > 0 else ""
        print(f"  {disease:25s} {count:5d}  {bar}")

    print(f"\n  {'TOTAL':25s} {total:5d}")
    print(f"\n💡 Minimum recommended: 100+ images per class (augmented to 500+)")
    print(f"   Consider supplementing with web-scraped images for sparse classes.")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Gerbera Disease Image Scraper")
    parser.add_argument(
        "--source",
        choices=["inaturalist", "gbif", "all"],
        default="all",
        help="Image source (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max images per source (default: 500)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🌻 GERBERA DISEASE IMAGE SCRAPER")
    print(f"   Output: {output_dir}")
    print(f"   Source: {args.source}")
    print(f"   Limit:  {args.limit} images per source")

    if args.source in ("inaturalist", "all"):
        scrape_inaturalist(output_dir, limit=args.limit)

    if args.source in ("gbif", "all"):
        scrape_gbif(output_dir, limit=args.limit)

    print_dataset_summary(output_dir)


if __name__ == "__main__":
    main()
