import streamlit as st
from PIL import Image
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from fastembed import TextEmbedding
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForZeroShotImageClassification
import google.generativeai as genai
import os
import numpy as np
import time

# Configure the Google Gemini API
os.environ['GOOGLE_API_KEY'] = st.secrets["GEMINI-PRO-API_KEY"]
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
vision_model = genai.GenerativeModel('gemini-1.5-flash')

# Load pre-trained models
model_name = "openai/clip-vit-base-patch32"
i_processor = AutoProcessor.from_pretrained(model_name)
i_model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)
embedding_model = TextEmbedding()
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize Qdrant client
api_key = st.secrets["qdrant_api_key"]
qdrant_url = st.secrets["q_url"]
client = QdrantClient(url=qdrant_url, api_key=api_key)

# Helper functions
def reduce_image_size(image, max_width=1024, max_height=1024):
    width, height = image.size
    scaling_factor = min(max_width / width, max_height / height)
    if scaling_factor < 1:
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        return resized_image
    return image

def embed_text(text):
    embed = embedding_model.embed([text])
    for i in embed:
        embed = i
    return embed

def generate_image_embedding(image):
    processed_img = i_processor(text=None, images=image, return_tensors="pt")['pixel_values']
    return i_model.get_image_features(processed_img).detach().numpy().tolist()[0]

def create_image_from_pixels(pixel_lst, img_size):
    pixel_lst = [tuple(p) if isinstance(p, list) else p for p in pixel_lst]
    img = Image.new('RGB', img_size)
    img.putdata(pixel_lst)
    return img

def process_image_and_text(image, text, retries=3):
    for attempt in range(retries):
        try:
            img_px = list(image.getdata())
            img_size = image.size

            text_embed = embed_text(text)
            text_point = PointStruct(id=str(uuid.uuid4()), vector=text_embed,
                                     payload={"pixel_lst": img_px, "img_size": img_size, "image_text": text})

            img_embed = generate_image_embedding(image)
            img_point = PointStruct(id=str(uuid.uuid4()), vector=img_embed,
                                    payload={"pixel_lst": img_px, "img_size": img_size, "image_text": text})

            client.upsert(collection_name="image_vectors", points=[img_point], wait=True)
            client.upsert(collection_name="text_vectors", points=[text_point], wait=True)

            st.success("Image and text have been processed and upserted to Qdrant!")
            break
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Retrying due to error: {e}. Attempt {attempt + 1}/{retries}")
                time.sleep(2)
            else:
                st.error(f"Failed after {retries} attempts: {e}")

def generate_image_description(image):
    try:
        response = vision_model.generate_content(["Describe the image content in 200 words", image])
        return response.text
    except Exception as e:
        st.error(f"Failed to generate description: {e}")
        return ""

# About page
def about_me():
        st.title('Ankit Jangid')
        st.write('Place: Jaipur, Rajasthan')
        st.write('Contact: +91 9461962044')

        # LinkedIn Profile Button
        if st.button('LinkedIn Profile'):
            st.markdown("[Open LinkedIn Profile](https://www.linkedin.com/in/ankit-jangid-9910b3231)", unsafe_allow_html=True)

        # Email Button
        if st.button('Mail'):
            st.markdown("[Send Email](mailto:ajladaniya425@gmail.com)", unsafe_allow_html=True)

        # Google Docs Resume Button
        if st.button('Google Doc'):
            st.markdown("[Open Google Docs](https://docs.google.com/document/d/1rzvM_XdlbO_hkSLwXwkVSb4imwsrKFNHrxA1LOKDNqA/edit?usp=sharing)", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Process", "Text Search", "Image Searching", "About"])

# Home page
if page == "Home":
    st.title("Image and Text Processing with Qdrant")
    st.markdown("""
    ### Welcome to the Image and Text Processing App!

    This application allows you to upload images with descriptions and store them in a Qdrant database.
    You can also search for images based on text or image queries.

    **Technologies Used:**
    - **Streamlit**: Web interface
    - **Qdrant**: Storing and searching embeddings
    - **FastEmbed**: Text embedding generation
    - **Gemini**: Image captioning
    """)

# Upload & Process page
elif page == "Upload & Process":
    st.title("Upload and Process Images and Texts")

    uploaded_images = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    auto_caption = st.checkbox("Automatically generate image captions")
    text_inputs = st.text_area("Enter descriptions for the images (one per line, sequence will be ↑)")

    if st.button("Process Images and Texts"):
        if uploaded_images:
            texts = text_inputs.split('\n') if not auto_caption else [generate_image_description(Image.open(img)) for img in uploaded_images]
            for i in range(len(texts)):
                st.write(f'Image {i+1} Description : {texts[i]}')
            if len(uploaded_images) != len(texts):
                st.error("The number of images and descriptions must match.")
            else:
                for image_file, text in zip(uploaded_images, texts):
                    image = Image.open(image_file)
                    resized_image = reduce_image_size(image)
                    st.image(resized_image, caption='Uploaded Image', use_column_width=True)
                    process_image_and_text(resized_image, text)
        else:
            st.error("Please upload images and enter descriptions.")

# Text Search page
elif page == "Text Search":
    st.title("Search for Similar Images")

    query = st.text_input("Enter a query to search for a similar image")
    if st.button("Search"):
        if query:
            try:
                embed = embed_text(query)
                search_result = client.search(collection_name="text_vectors", query_vector=embed, limit=1)
                if search_result:
                    payload = search_result[0].payload
                    pixel_lst = payload["pixel_lst"]
                    img_size = payload["img_size"]
                    img = create_image_from_pixels(pixel_lst, tuple(img_size))
                    st.image(img, caption='Retrieved Image', use_column_width=True)
                else:
                    st.error("No matching image found.")
            except Exception as e:
                st.error(f"An error occurred during the search: {e}")
        else:
            st.error("Please enter a query.")

# Image Searching page
elif page == "Image Searching":
    query_images = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Image Search"):
        if query_images:
            for query_image in query_images:
                try:
                    image = Image.open(query_image)
                    resized_image = reduce_image_size(image)
                    img_embeddings = generate_image_embedding(resized_image)
                    hits = client.search(collection_name="image_vectors", query_vector=img_embeddings, limit=1)
                    if hits:
                        hit = hits[0]
                        img_size = tuple(hit.payload['img_size'])
                        pixel_lst = hit.payload['pixel_lst']
                        img = create_image_from_pixels(pixel_lst, img_size)
                        st.image(img, caption='Retrieved Image', use_column_width=True)
                    else:
                        st.error("No matching image found.")
                except Exception as e:
                    st.error(f"An error occurred during the search: {e}")
        else:
            st.error("Please upload an image to search.")

# About page
elif page == "About":
    about_me()
    
