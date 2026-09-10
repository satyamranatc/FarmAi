import os
import io
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

@st.cache_resource
def get_model():
    return load_model("model.h5")

model = get_model()
classNames = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']

def soliTypeFinder(soilPic):
    # Load new image
    image = load_img(
        io.BytesIO(soilPic.getvalue()),
        target_size=(128, 128)
    )

    # Convert image to numbers
    image = img_to_array(image)

    # Add one extra dimension
    image = np.expand_dims(image, axis=0)

    # Ask CNN to predict
    prediction = model.predict(image)

    # Find highest probability
    result = np.argmax(prediction)
    return classNames[result]


def askAi(**farmDetails):
    if not client:
        return "Error: GROQ_API_KEY is not set. Please provide a valid Groq API key in your .env file."

    prompt = f"""
You are a master farmer with 20+ years of experience in farming worldwide.
You have deep expertise in farming with different types of soil and crops across the globe.
You understand soil and crop types and their properties and crop yield with factors like climate, farm size, and farming budget.

Now analyze this farm:
- SOIL TYPE: {farmDetails.get("soilType", "Unknown")}
- CLIMATE: {farmDetails.get("climate", "Unknown")}
- FARM SIZE: {farmDetails.get("farmSize", "Unknown")} ACRES
- BUDGET: {farmDetails.get("budget", "Unknown")} INR
- FARMER NAME: {farmDetails.get("userName", "Farmer")}

Guide {farmDetails.get("userName", "the farmer")} with actionable, step-by-step advice on:
1. Best suited crops for this soil, climate, and budget
2. Crop rotation and soil enrichment tips
3. Estimated timeline and yield expectations
4. Budget allocation and cost-saving techniques
"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="openai/gpt-oss-120b",
    )

    return chat_completion.choices[0].message.content


st.set_page_config(page_title="Farm AI", page_icon="🌱", layout="centered")
st.title("🌱 Farm AI - Smart Crop & Soil Advisor")
st.write("Upload an image of your farm's soil and enter your details to receive AI-powered farming guidance.")

farmSoilpic = st.file_uploader("Upload an image of soil", type=["jpg", "png", "jpeg"])
if farmSoilpic is not None:
    st.image(farmSoilpic, caption="Uploaded Soil Image", width=250)

userName = st.text_input("Enter your name")
farmSize = st.number_input("Enter your farm size in acres", min_value=0.1, step=0.5)
climate = st.text_input("Enter your farm climate (e.g. Tropical, Arid, Moderate)")
budget = st.number_input("Enter your budget in INR", min_value=1000, step=5000)

if st.button("Analyze & Get Farming Advice"):
    if farmSoilpic is not None and userName and farmSize and climate and budget:
        with st.spinner("Analyzing soil type..."):
            soilType = soliTypeFinder(farmSoilpic)
            st.success(f"Detected Soil Type: **{soilType}**")

        with st.spinner("Generating personalized AI farming plan..."):
            res = askAi(soilType=soilType, userName=userName, farmSize=farmSize, climate=climate, budget=budget)
            st.subheader("🌾 AI Farming Recommendation")
            st.markdown(res)
    else:
        st.warning("Please upload a soil image and fill in all fields before submitting.")


