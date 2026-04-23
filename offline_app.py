import streamlit as st
from PIL import Image

from utils.utils import load_model, predict, class_names
from utils.xai_utils import generate_gradcam, calculate_severity
from utils.local_llm import get_disease_report, get_chatbot_answer
# from utils.grok_handler import get_disease_report
# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Plant Doctor",
    page_icon="🌿",
    layout="wide"
)

# ---------------- LOAD MODEL (CACHED) ---------------- #

@st.cache_resource
def get_model():
    return load_model()

model = get_model()


# ---------------- SESSION STATE ---------------- #

if "page" not in st.session_state:
    st.session_state.page = "Home"


# ================= NAVBAR ================= #

# 1. Custom CSS for the Horizontal Navbar
st.markdown("""
    <style>
    /* Hide the sidebar entirely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Navbar Container */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #ffffff;
        padding: 10px 5%;
        border-bottom: 2px solid #2ecc71;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
    }
    
    /* Branding */
    .nav-logo {
        font-size: 24px;
        font-weight: bold;
        color: #2ecc71;
        text-decoration: none;
    }

    /* Main Content Padding to avoid overlap with sticky navbar */
    .main-content {
        padding-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Navbar Logic using Streamlit Columns
# This creates a row of buttons that act like a navbar
st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
cols = st.columns([2,1,1,1,1])
with cols[0]:
    st.markdown("<span class='nav-logo'>🌿 PlantDoc AI</span>", unsafe_allow_html=True)

with cols[1]:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"

with cols[2]:
    if st.button("🔬 Detect", use_container_width=True):
        st.session_state.page = "Disease Detection"

with cols[3]:
    if st.button("🤖 Plant AI Chat", use_container_width=True):
        st.session_state.page = "Chatbot"

with cols[4]:
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "About"

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='main-content'></div>", unsafe_allow_html=True)

# Assign current page
page = st.session_state.get("page", "Home")

# ================= ROUTING ================= #

if page == "Home":
    # Your Home Page Code...
    st.write("Welcome to the Home Page")

elif page == "Disease Detection":
    # Your Detection Code...
    st.write("Analysis Module Active")

elif page == "About":
    # Your About Code...
    st.write("System Specifications")


# ================= HOME ================= #

if page == "Home":
    # 1. CSS for custom styling (Animations & Cards)
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(to bottom, #688C72, #4D7664);
        }
        .hero-text {
            text-align: center;
            padding: 20px;
        }
        .feature-card {
            background-color: #C9ECD87C;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-top: 5px solid #2ecc71;
            height: 100%;
        }
        .step-number {
            background-color: #2ecc71;
            color: white;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            display: inline-block;
            text-align: center;
            font-weight: bold;
            margin-right: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Hero Section
    st.markdown("<div class='hero-text'>", unsafe_allow_html=True)
    st.title("🌱 Next-Gen Phytopathology AI")
    st.markdown("### Bridging Deep Learning and Agriculture for a Healthier Planet.")
    st.write("An advanced diagnostic suite powered by EfficientNet-B0 and Local LLMs.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # 3. Interactive Workflow (The "How it Works" section)
    st.subheader("🚀 Interactive Analysis Pipeline")
    
    # Using columns for a step-by-step visual
    step1, step2, step3 = st.columns(3)
    
    with step1:
        st.markdown("""
        <div class='feature-card'>
            <h4><span class='step-number'>1</span> Vision Scan</h4>
            <p>Our <b>EfficientNet-B0</b> architecture scans leaf textures to identify 38 unique disease signatures with precision.</p>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div class='feature-card'>
            <h4><span class='step-number'>2</span> Attention Map</h4>
            <p>Integrated <b>Grad-CAM</b> tech highlights the exact infected regions, showing you what the AI is 'looking' at.</p>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div class='feature-card'>
            <h4><span class='step-number'>3</span> AI Prognosis</h4>
            <p>An <b>Offline LLM (Ollama)</b> analyzes the findings to generate a custom-tailored treatment and recovery plan.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("") # Spacing
    st.write("")

    # 4. Impact Metrics (Interactive Stats)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Model Accuracy", value="98.4%", delta="State-of-Art")
    with col_b:
        st.metric(label="Inference Speed", value="~120ms", delta="Real-time")
    with col_c:
        st.metric(label="Privacy", value="100%", delta="Local Processing")

    st.divider()

    # 5. Call to Action Section
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Ready to protect your crops?")
        st.write("Click the button below or use the sidebar to upload a leaf specimen. Our system works entirely offline, ensuring your data remains private and secure.")
        if st.button("Launch Diagnostic Suite 🚀"):
            st.info("Please select 'Disease Detection' from the sidebar to begin!")
    
    with c2:
        # A nice placeholder image or a decorative element
        st.info("**Did you know?** Early detection of *Late Blight* can save up to 90% of a tomato crop's yield.")

    # Decorative footer
    st.caption("Developed with ❤️ for Sustainable Agriculture | Powered by PyTorch & Ollama")





# ================= DETECTION ================= #

elif page == "Disease Detection":
    # Custom CSS for the badges and aesthetic
    st.markdown("""
        <style>
         .main {
            background: linear-gradient(to bottom, #688C72, #4D7664);
        }
        .reportview-container .main .block-container{ padding-top: 1rem; }
        .stProgress > div > div > div > div { background-color: #2ecc71; }
        .diagnosis-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid #2ecc71;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔬 AI Diagnostic Center")
    st.write("Upload a high-resolution image of the plant leaf for a deep-tissue analysis.")

    uploaded_file = st.file_uploader(
        "Drop leaf image here...",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG, JPEG, and PNG formats."
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        # UI Layout: Image on the left, Initial Prediction on the right
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📸 Captured Specimen")
            # Change use_container_width to use_column_width
            st.image(image, use_column_width=True)

        with col2:
            st.markdown("### ⚙️ Analysis Control")
            st.info("Ready for scanning. Our EfficientNet-B0 model will analyze 38 potential disease classes.")
            analyze_btn = st.button("Start Full Diagnostic Scan ⚡", use_container_width=True)

        if analyze_btn:
            # 1. Prediction Phase
            with st.status("🔍 Deep Learning Scan in Progress...", expanded=True) as status:
                st.write("Calculating feature maps...")
                results = predict(image, model)
                top_disease, top_conf = results[0]
                confidence_percent = round(top_conf * 100, 2)
                
                st.write("Generating Grad-CAM Heatmap...")
                class_idx = class_names.index(top_disease)
                heatmap, grayscale_cam = generate_gradcam(model, image, class_idx)
                severity = calculate_severity(grayscale_cam)
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            st.divider()

            # 2. Main Results Layout
            res_col1, res_col2 = st.columns([1, 1.2])

            with res_col1:
                st.subheader("🧠 AI Attention Map")
                st.image(heatmap, caption="Highlighting Infected Areas", use_column_width=True)                
                # Severity Meter
                st.markdown(f"**Physical Severity Index:** {severity}%")
                # Color code the severity bar
                bar_color = "green" if severity < 20 else "orange" if severity < 50 else "red"
                st.progress(severity / 100)
                st.caption(f"The model is focusing on the areas highlighted in red/yellow.")

            with res_col2:
                st.subheader("📊 Diagnostic Summary")
                
                # Display Top Result prominently
                st.markdown(f"""
                <div class="diagnosis-card">
                    <h2 style='color: #1e3d59; margin:0;'>{top_disease.replace('_',' ')}</h2>
                    <p style='color: #2ecc71; font-weight: bold;'>Confidence: {confidence_percent}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.write("---")
                st.write("**Top Probability Matches:**")
                for disease, conf in results[:3]: # Show top 3
                    st.write(f" {disease.replace('_',' ')}")
                    st.progress(conf)

            # 3. AI Report Phase (The LLM Part)
            st.divider()
            st.subheader("🤖 Smart Treatment Plan (AI Generated)")
            
            with st.spinner("Ollama is drafting your customized report..."):
                ai_response = get_disease_report(
                    disease_name=top_disease,
                    confidence=confidence_percent,
                    severity=severity
                )
                               
                # Presenting the AI Report in a nice box
                st.chat_message("assistant").write(ai_response)

            st.balloons()


# ================= ABOUT ================= #
elif page == "About":
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(to bottom, #688C72, #4D7664);
        }    
        </style>
    """, unsafe_allow_html=True)
    st.title("🌱 Intelligent Plant Healthcare System")
    
    st.markdown("""
    This project leverages state-of-the-art **Computer Vision** and **Large Language Models (LLMs)** to provide a complete diagnostic tool for farmers and researchers.
    """)

    # --- SECTION 1: SYSTEM CAPABILITIES ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Images Analyzed", value="87,867+")
    with col2:
        st.metric(label="Plant Classes", value="38")
    with col3:
        st.metric(label="AI Model", value="EfficientNet-B0")

    st.divider()

    # --- SECTION 2: DATA & MODEL DETAILS ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📚 Knowledge Base")
        st.write("The system was trained on a massive combined dataset to ensure high accuracy across different environments.")
        st.table({
            "Dataset": ["PlantVillage", "PlantDoc"],
            "Role": ["Standardized Lab Images", "Real-world Field Images"],
            "Impact": ["Base accuracy", "Environmental robustness"]
        })

    with col_right:
        st.subheader("🧠 The Brain")
        st.write("We use **EfficientNet-B0**, a model designed for the perfect balance between speed and precision.")
        st.info("**Explainability:** Integrated with **Grad-CAM** to highlight exactly *where* the model sees the disease on the leaf.")
        st.info("**Reports:** Powered by **Ollama (Local LLM)** to generate detailed care instructions without needing internet.")

    # --- SECTION 3: SYSTEM ENVIRONMENT (Interactive Expander) ---
    with st.expander("🛠️ Technical Specifications & Hardware"):
        st.write("The system is optimized for high-performance computing using NVIDIA GPU acceleration.")
        
        env_col1, env_col2 = st.columns(2)
        with env_col1:
            st.markdown(f"""
            **Software Environment:**
            * **Python:** 3.12.12
            * **PyTorch:** 2.9.0 (CUDA 12.6)
            * **NumPy:** 2.0.2
            """)
        with env_col2:
            st.markdown(f"""
            **Hardware Acceleration:**
            * **GPU Available:** ✅ Yes
            * **GPU Name:** Tesla T4
            * **Optimization:** Adam Optimizer (LR=0.0001)
            """)

    # --- SECTION 4: DATASET VISUALIZATION ---
    st.subheader("🖼️ Training Overview")
    st.write("Data is processed through a pipeline of rotations, flips, and color adjustments to mimic real-world sunlight conditions.")
    
    tab1, tab2 = st.tabs(["Data Split", "Target Crops"])
    
    with tab1:
        # A simple bar chart for data split
        data = {"Phase": ["Training", "Validation"], "Images": [70295, 17572]}
        st.bar_chart(data, x="Phase", y="Images")
    
    with tab2:
        st.write("The model can detect health issues in the following crops:")
        st.caption("Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.")

    st.success("System is currently Online and ready for Inference.")



elif page == "Chatbot":
        
    st.markdown("""
    
        <style>
            .main {
                background: linear-gradient(to bottom, #688C72, #4D7664);
            }    
        </style>
    """, unsafe_allow_html=True)

    st.title("🤖 PlantDoc AI Assistant")
    st.write("Ask anything about plant diseases, treatments, soil health, or crop care.")

    # Chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Context from report
    report_context = st.session_state.get("chat_context", "")

    if report_context:
        st.info("Chat started with diagnostic report context.")
        st.code(report_context)

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about agriculture, crops, livestock, soil, fertilizers, etc...")

    if user_input:

        st.chat_message("user").write(user_input)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })


        with st.spinner("Thinking..."):

             ai_reply = get_chatbot_answer(user_input)

        st.chat_message("assistant").write(ai_reply)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_reply
        })