def generate_ai_prompt(disease_name, confidence, severity):

    prompt = f"""
You are an expert plant pathologist.

A deep learning model detected the following:

Disease Name: {disease_name}
Model Confidence: {confidence*100:.2f}%
Estimated Severity: {severity:.2f}%

Provide:

1. Detailed explanation of the disease
2. Causes
3. Symptoms
4. Prevention methods
5. Recommended treatment (organic and chemical)
6. Farmer-friendly advice

Keep the response clear and practical.
"""

    return prompt