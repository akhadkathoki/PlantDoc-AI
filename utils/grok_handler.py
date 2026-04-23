import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

# Get API key from .env
API_KEY = os.getenv("GROQ_API_KEY")

# Create Groq client using the API key
client = Groq(api_key=API_KEY)


def get_disease_report(disease_name, confidence, severity):

    prompt = f"""
    You are an expert agricultural plant pathologist.

    Disease: {disease_name}
    Confidence: {confidence}%
    Severity: {severity}%

    Provide:
    1. Description
    2. Causes
    3. Treatment
    4. Prevention
    5. Farmer Advice
    """

    try:
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return chat.choices[0].message.content

    except Exception as e:
        return f"⚠️ Groq API Error: {str(e)}"
    

def get_chatbot_answer(user_input):

    prompt = f"""
You are **PlantDoc AI**, a professional agricultural assistant.

You can ONLY answer questions related to the agriculture sector including:

• Plant diseases
• Crop farming
• Soil health and soil improvement
• Fertilizers and pesticides
• Irrigation methods
• Agricultural tools and instruments
• Livestock and domestic farm animals
• Sustainable farming
• Agricultural technology
• Crop yield improvement
• Organic farming

If the user asks something OUTSIDE agriculture, politely refuse and say:

"I'm designed to assist only with agriculture related topics such as farming, crops, soil health, livestock, and agricultural technology."

User Question:
{user_input}

Provide clear, practical, farmer-friendly advice.
"""

    try:
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return chat.choices[0].message.content

    except Exception as e:
        return f"⚠️ Chatbot Error: {str(e)}"