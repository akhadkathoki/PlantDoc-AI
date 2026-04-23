import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def get_disease_report(disease_name, confidence, severity):

    prompt = f"""
You are an expert agricultural plant pathologist.

Disease Name: {disease_name}
Model Confidence: {confidence}%
Severity Level: {severity}%

Provide:
1. Disease Description
2. Causes
3. Treatment
4. Prevention
5. Farmer Advice

Keep response structured and clear.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "phi3:mini",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
    )

    data = response.json()

    return data["message"]["content"]



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

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "phi3:mini",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
    )

    data = response.json()

    return data["message"]["content"]



