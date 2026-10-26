import pandas as pd
import os
import re
from pathlib import Path
from urllib.parse import urlparse

# CLIP imports
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Dataset location
ROOT = Path(
    r"C:\Users\user\OneDrive\Documents\IMMFND_Cleaned"
)


# 1. Basic source credibility heuristic
def credibility_score(url):
    """Heuristic score only; not a verified trust rating."""

    if pd.isna(url) or not str(url).strip():
        return 0.3

    domain = urlparse(str(url)).netloc.lower()
    domain = domain.removeprefix("www.")

    if not domain:
        return 0.3

    known_domains = {
    "reuters.com": 0.8,
    "bbc.com": 0.8,
    "bbc.co.uk": 0.8,
    "apnews.com": 0.8,
    "abplive.com": 0.5,
    "thestatesman.com": 0.5,
    "tribuneindia.com": 0.5,
    "vishvasnews.com": 0.5,
}

    return known_domains.get(domain, 0.5)


# 2. Preliminary evidence classification
def classify_evidence(claim, evidence, similarity):
    """
    Labels evidence relevance using keyword overlap.
    Does not establish whether a claim is true or false.
    """

    if pd.isna(claim) or pd.isna(evidence):
        return "UNCERTAIN"

    if similarity is None or pd.isna(similarity):
        return "UNCERTAIN"

    if similarity < 0.15:
        return "IRRELEVANT"

    claim_words = set(re.findall(r"\w+", str(claim).lower()))
    evidence_words = set(re.findall(r"\w+", str(evidence).lower()))

    if not claim_words or not evidence_words:
        return "UNCERTAIN"

    overlap = len(claim_words & evidence_words) / len(claim_words)

    if overlap >= 0.5:
        return "POSSIBLY_RELATED"

    return "UNCERTAIN"


# 3. Retrieve related articles using keyword overlap

def retrieve_evidence(query, df, top_k=5):
    """Retrieve candidate articles using normalized keyword overlap."""

    if pd.isna(query) or not str(query).strip():
        return pd.DataFrame()

    def tokenize(text):
        text = str(text).lower()
        return {
            word.strip(".,!?;:()[]{}\"'।")
            for word in text.split()
            if len(word.strip(".,!?;:()[]{}\"'।")) > 1
        }

    query_words = tokenize(query)

    if not query_words:
        return pd.DataFrame()

    results = []

    for _, row in df.iterrows():
        title = "" if pd.isna(row.get("title")) else str(row.get("title"))
        claim = "" if pd.isna(row.get("claim")) else str(row.get("claim"))
        content = "" if pd.isna(row.get("content")) else str(row.get("content"))

        text = f"{title} {claim} {content}"
        article_words = tokenize(text)

        overlap = len(query_words & article_words)
        score = overlap / len(query_words)

        if score >= 0.3:
            url = row.get("url", "")

            results.append({
    "title": title,
    "url": url,
    "image_src": row.get("image_src", None),
    "dataset_label": row.get("label", "unknown"),
    "retrieval_score": round(score, 3),
    "source_score": credibility_score(url),
    "evidence_label": classify_evidence(query, text, score)
})

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values("retrieval_score", ascending=False)
        .head(top_k)
    )



# 4. Load the real dataset

real_file = ROOT / "train" / "Real" / "Real_text.xlsx"
fake_file = ROOT / "train" / "Fake" / "Fake_text.xlsx"

real_df = pd.read_excel(real_file)
fake_df = pd.read_excel(fake_file)

df = pd.concat(
    [real_df, fake_df],
    ignore_index=True
)

print("Real articles:", len(real_df))
print("Fake articles:", len(fake_df))
print("Combined dataset:", len(df), "rows")
print("Dataset labels:")
print(df["label"].value_counts().to_string())

print("Dataset loaded:", len(df), "rows")


# 5. Test evidence classification
print("\n--- Evidence Classification Tests ---")

print(
    "Related:",
    classify_evidence(
        "India launches new solar energy project",
        "India launches new solar energy project today",
        0.8
    )
)

print(
    "Irrelevant:",
    classify_evidence(
        "India launches new solar energy project",
        "A football team won its match",
        0.05
    )
)

print(
    "Uncertain:",
    classify_evidence(
        "India launches new solar energy project",
        "Weather conditions changed across several regions",
        0.8
    )
)


# 6. Test retrieval using an actual dataset claim
sample_claim = str(
    df[df["label"] == "fake"].iloc[0]["claim"]
)

print("\n--- Evidence Retrieval Test ---")
print("Search claim:", sample_claim)

evidence = retrieve_evidence(
    sample_claim,
    df,
    top_k=5
)

if evidence.empty:
    print("No matching evidence found.")
else:
    print(
    evidence[
        [
            "title",
            "url",
            "dataset_label",
            "retrieval_score",
            "source_score",
            "evidence_label"
        ]
    ].to_string(index=False)
)

# 7. Test source scores on five dataset records
print("\n--- Source Credibility Tests ---")

for _, row in df.head(5).iterrows():
    url = row.get("url", "")
    claim = row.get("claim", "")

    print("\nClaim:", str(claim)[:120])
    print("URL:", url)
    print("Source credibility heuristic:", credibility_score(url))
print("\n--- Known Domain Test ---")
print("Reuters:", credibility_score("https://www.reuters.com/world/"))
print("BBC:", credibility_score("https://www.bbc.com/news"))
print("Unknown website:", credibility_score("https://example.com"))
# --- CLIP Image-Text Similarity ---

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

clip_model = None
clip_processor = None


def load_clip_model():
    global clip_model, clip_processor

    if clip_model is None:
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    return clip_model, clip_processor



def compare_image_text(image_url, descriptions):
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")

        print("DEBUG 1: Loading CLIP model...", flush=True)
        model, processor = load_clip_model()
        print("DEBUG 2: CLIP model loaded.", flush=True)

        inputs = processor(
            text=descriptions,
            images=image,
            return_tensors="pt",
            padding=True
        )
        print("DEBUG 3: Image and text processed.", flush=True)

        with torch.no_grad():
            outputs = model(**inputs)

        print("DEBUG 4: CLIP comparison completed.", flush=True)

        scores = outputs.logits_per_image[0].tolist()

        results = sorted(
            zip(descriptions, scores),
            key=lambda item: item[1],
            reverse=True
        )

        return results

    except Exception as error:
        print("Image comparison failed:", error)
        return []

print("\nBasic prototype tests complete.")


# --- Combined Research Prototype Report ---

print("\n" + "=" * 55)
print("COMBINED RESEARCH PROTOTYPE REPORT")
print("=" * 55)

print("\n1. CLAIM")
print(sample_claim)

print("\n2. RETRIEVED EVIDENCE")

if evidence is not None and not evidence.empty:
    for index, row in evidence.iterrows():
        print(f"\nEvidence {index + 1}:")
        print("Title:", row["title"])
        print("Dataset label:", row["dataset_label"])
        print("Retrieval score:", round(row["retrieval_score"], 4))
        print("Source heuristic score:", row["source_score"])
        print("Evidence relevance label:", row["evidence_label"])
        print("URL:", row["url"])
        print("Image URL:", row.get("image_src"))
else:
    print("No matching evidence found.")

print("\n3. IMAGE-TEXT SIMILARITY")

# Get the image URL from the dataset for the sample claim
sample_rows = df[df["claim"] == sample_claim]

test_image_url = None

if evidence is not None and not evidence.empty:
    image_value = evidence.iloc[0].get("image_src")

    if pd.notna(image_value) and str(image_value).startswith(("https://", "http://")):
        test_image_url = str(image_value)

print("Retrieved evidence image URL:", test_image_url)

test_descriptions = [
    "An aerial photograph of a heart-shaped pond in Crimea.",
    "An aerial photograph of a circular lake.",
    "An aerial photograph of a heart-shaped pond in Russia.",
    "An aerial photograph of an unrelated landscape."
]

image_results = compare_image_text(test_image_url, test_descriptions)
print("\nDEBUG: Number of image results:", len(image_results))

if image_results:
    best_description, best_score = image_results[0]

    print("\nHighest-scoring image description:")
    print(f"Description: {best_description}")
    print(f"Similarity score: {best_score:.4f}")

    print("\nAll image-text similarity results:")
    for description, score in image_results:
        print(f"Description: {description}")
        print(f"Similarity score: {score:.4f}")
else:
    print("No image-text similarity results available.")



print("\n4. RESEARCH LIMITATIONS")
print("- Retrieval scores indicate keyword overlap.")
print("- Source scores are manually assigned heuristics.")
print("- Relevance labels do not prove support or contradiction.")
print("- CLIP similarity does not establish claim truth.")
print("\nCombined report complete.")
