import os
import cv2
import pytesseract
import faiss
import numpy as np
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer
import easyocr

# Setup Google Vision API client
credentials = service_account.Credentials.from_service_account_file(
    'Use your Google Cloud credentials here'
)
client = vision.ImageAnnotatorClient(credentials=credentials)

# Load a pre-trained model for text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from an image using Tesseract OCR


def extract_text(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path, detail=0)
    return ' '.join(result)

# Generate image descriptions using Google Vision API
def generate_image_description(image_path):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = [label.description for label in response.label_annotations]
    return ', '.join(labels)

# Create a vector index using FAISS
def create_faiss_index(descriptions):
    dimension = 384  # Dimension of the embedding model
    index = faiss.IndexFlatL2(dimension)
    vectors = [desc["vector"] for desc in descriptions]
    # Normalize vectors
    vectors = np.array([v / np.linalg.norm(v) for v in vectors], dtype=np.float32)
    index.add(vectors)
    return index

# Search for relevant memes
# ... existing code ...

# Search for relevant memes
def search_meme(index, descriptions, query, top_k=5):
    query_vector = model.encode(query)
    # Normalize query vector
    query_vector = query_vector / np.linalg.norm(query_vector)
    distances, indices = index.search(np.array([query_vector], dtype=np.float32), len(descriptions))
    results = [(descriptions[i]["image_path"], distances[0][i]) for i in indices[0]]
    # Directly take the top_k results without sorting
    top_results = results[:top_k]
    return top_results

# Main Execution Flow
def main(image_folder, query):
    descriptions = []

    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        text = extract_text(img_path)
        description = generate_image_description(img_path)
        full_desc = f"{description}. Text: {text}"

        # Generate vector embedding for the description
        embedding = model.encode(full_desc)

        descriptions.append({
            "image_path": img_path,
            "description": full_desc,
            "vector": embedding,
        })

    # Build and search the FAISS index
    index = create_faiss_index(descriptions)
    results = search_meme(index, descriptions, query)

    for img_path, score in results:  # Updated to unpack two values
        print(f"Found: {img_path} (Score: {score:.2f})")

# ... existing code ...


if __name__ == "__main__":
    image_folder = "memes"
    while True:
        query = input("Enter your query (type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        main(image_folder, query)