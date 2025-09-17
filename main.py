import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Gemini Initialization (Confirmed)
import os
import google.generativeai as genai





# Set page configuration - this should be the very first Streamlit command
st.set_page_config(
    page_title="Plant Disease Recognition", 
    page_icon="🌱", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# #----------------------------------------------------- Speech Feature ---------------------------------------------------------------------------------
# # Initialize the TTS engine
# engine = pyttsx3.init()

# # Function to speak the result using TTS in a separate thread
# def speak(text):
#     def run():
#         engine.say(text)
#         engine.runAndWait()
    
#     # Run the speech in a background thread
#     thread = threading.Thread(target=run)
#     thread.start()
# #----------------------------------------------------- End Speech Feature ---------------------------------------------------------------------------------


# Load the model once at the start
model = tf.keras.models.load_model("maize_disease_detection_model.h5", compile=False)

#Tensorflow Model Prediction
def model_prediction(test_image):
# #    model = tf.keras.models.load_model("maize_disease_detection_model.h5")

#     image = tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))
#     # processing image ka cusub ee shaqayna
#     image_arr = tf.keras.preprocessing.image.img_to_array(image)
#     # new convert single image to batch (waxa u ku haboon yahay modelkena)    
#     input_arr = np.expand_dims(image_arr, axis=0)
#     # Predicting the image 
#     predictions = model.predict(input_arr)
#     # kalsoni inte leeg ayuu ku qabaa saadaalintan 
#     confidence_pct = round(100 * (np.max(predictions)), 2)
#     #return index of max element
#     return np.argmax(predictions), confidence_pct 

    """
    Somali: Tani waxay aqbashaa UploadedFile (upload ama kamarad), bytes, path, ama PIL.Image.
    Waxay u beddeshaa PIL.Image -> (256,256) -> array -> model.predict.
    Waxay soo celineysaa (result_index:int, confidence_pct:float).
    """
    import io  # Somali: Si aan bytes/file-like ugu rogno sawir
    from PIL import Image

    # Somali: Akhri sawirka si badbaado leh iyadoo laga yaabo inuu ka yimid kamarad/upload/path
    try:
        if hasattr(test_image, "read"):          # UploadedFile / file-like (upload ama kamarad)
            test_image.seek(0)
            pil_img = Image.open(test_image).convert("RGB")
        elif isinstance(test_image, (bytes, bytearray)):  # bytes
            pil_img = Image.open(io.BytesIO(test_image)).convert("RGB")
        else:  # path ama wax kale
            pil_img = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
            pil_img = pil_img.convert("RGB")
    except Exception as e:
        # Somali: Haddii akhrisku khaldamo, fariin kooban
        raise ValueError("Sawirka lama akhrin karo. Fadlan mar kale isku day.") from e

    pil_img = pil_img.resize((256, 256))  # Somali: Hubi cabbirka model-ka
    image_arr = tf.keras.preprocessing.image.img_to_array(pil_img)
    input_arr = np.expand_dims(image_arr, axis=0)

    # Somali: Isticmaal model-kaaga la soo rartay sida uu koodhkaaga hore u yahay
    predictions = model.predict(input_arr)
    confidence_pct = round(100 * float(np.max(predictions)), 2)
    return int(np.argmax(predictions)), confidence_pct
# halkan waa halkaan ku joogeye.



# Sidebar Styling
st.sidebar.markdown(
    f"""
    <style>
    /* Sidebar Styles */
    .sidebar .sidebar-title {{
        color: #333333;  /* Dark gray for the title */
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 20px 0;
    }}
    .sidebar hr {{
        border-top: 2px solid #cccccc;  /* Light gray for the divider */
        margin: 10px 0;
    }}
    
    /* Custom Styling for Radio Buttons */
    .sidebar .stRadio {{
        background-color: #f4f4f4;  /* Light gray for the background */
        color: #333333;  /* Dark gray text */
        padding: 10px;
        border-radius: 5px;
    }}
    .sidebar .stRadio label {{
        font-size: 1.2em;
        color: #333333;
        margin-bottom: 15px;
    }}
    .sidebar .stRadio input {{
        background-color: #f4f4f4;
        color: #333333;
        border: 1px solid #cccccc;
        padding: 5px;
        border-radius: 5px;
        font-size: 1.1em;
    }}
    .sidebar .stRadio input:checked {{
        border-color: #888888;  /* Darker border color when checked */
    }}
    .sidebar .stRadio input:focus {{
        border-color: #444444;  /* Focused border color */
    }}
    .sidebar .stRadio input:hover {{
        background-color: #eaeaea;  /* Slight hover effect for inputs */
    }}

    /* Customize the active option */
    .sidebar .stRadio input:checked + label {{
        color: #0066cc;  /* Blue for selected option */
        font-weight: bold;
    }}

    /* Styling for each page in the sidebar */
    .sidebar .stRadio div {{
        padding: 8px;
        border-radius: 5px;
    }}

    </style>
    """, 
    unsafe_allow_html=True
)

# Sidebar with Styled Radio Buttons and Contact Option
st.sidebar.title("🌿 Dashboard")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# Radio buttons for page selection, now with Contact page added
app_mode = st.sidebar.selectbox(
    "Select a Page:",
    ["🏠 Home", "📖 About", "🌿 Disease Detection", "📞 Contact", "🤖 AI Chat"],  # Added "Contact"
    index=0  # Default selection
)



# Based on the radio selection, display the corresponding page
if app_mode == "🏠 Home":
    st.write("")
elif app_mode == "📖 About":
    st.write("About Us")
elif app_mode == "🌿 Disease Detection":
    st.write("Disease Detection Page")
elif app_mode == "🤖 AI Chat":
    st.write("Somali AI Agriexpert")
elif app_mode == "📞 Contact":
    st.header("📞Contact Us")


# ----------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------start-contact--------------------------------------------------------------------------
   
# Create two columns: left for form, right for contact info
    col1, col2 = st.columns([1, 1])  # Equal width columns
    
    with col1:
                # Contact Form
        contact_name = st.text_input("Your Name", placeholder="Enter your full name")
        contact_email = st.text_input("Your Email", placeholder="Enter your email address")
        contact_message = st.text_area("Your Message",  placeholder="Write your message here", height=150)
        
        if st.button("Send Message"):
            if contact_name and contact_email and contact_message:
                st.success("Message sent successfully! 🎉")
            else:
                st.error("Please fill in all fields.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Contact Information
        st.subheader("Our Contact Information")
        st.markdown('<div class="contact-info">', unsafe_allow_html=True)
        st.markdown(
            """
            <ul>
                <li>📧 Email: contact@plantdisease.com</li>
                <li>📞 Phone: +1234567890</li>
                <li>📍 Address: 123 Green Street, Agriculture City, Farm Country</li>
            </ul>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)



# Define Colors
primary_green = "#6A994E"
accent_yellow = "#FFD166"
warm_brown = "#A56D42"
soft_beige = "#F5F5DC"
dark_olive_green = "#2D4739"
text_gray = "#4F4F4F"
secondary_light_green = "#B7E4C7"
# ----------------------------------------------------------------------------------------------------------------------------------------



#------------------------------------------------ Home Page Content----------------------------------------------------
if app_mode == "🏠 Home":
    st.markdown(
    f"""
    <style>
    .main-title {{
        text-align: center;
        font-size: 2.5em;
        color: {primary_green};
    }}
    .home-content {{
        text-align: center;
        font-size: 1.2em;
        line-height: 1.6;
        color: {text_gray};
    }}
    .feature-card {{
        background: linear-gradient(135deg, #3726a6, #06d063);
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        text-align: center;
        height: 200px;  /* Set the height of the cards to be the same */
        display: flex;
        flex-direction: column;
        justify-content: center;  /* Align content vertically */
        color: white;  /* Ensure text is readable */
    }}
    .feature-card h3 {{
        font-size: 1.5em;
    }}
    .btn {{
        background-color: {accent_yellow};
        color: {dark_olive_green};
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }}
    .btn:hover {{
        background-color: {warm_brown};
    }}
    .stat-card {{
        background-color: {soft_beige};
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
    }}
    .stat-card h3 {{
        font-size: 1.2em;
        color: {primary_green};
    }}
    a {{
        color: {primary_green};
        text-decoration: none;
    }}
    a:hover {{
        color: {accent_yellow};
        text-decoration: underline;
    }}
    </style>
    """, 
    unsafe_allow_html=True
)


# Page title
    st.markdown("<h1 class='main-title'>🌱 Plant Disease Recognition System 🌱</h1>", unsafe_allow_html=True)

# Home content
    st.markdown(
    """
    <div class="home-content">
        <p>Welcome to the Plant Disease Recognition System! Our mission is to empower farmers by providing a simple, 
        efficient, and reliable way to identify maize leaf diseases early. <strong> Early detection can save crops, improve yield, and contribute to sustainable agriculture.</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Styled image display with animation
    st.markdown(
    """
    <style>
    @keyframes fadeInZoom {
        0% {
            opacity: 0;
            transform: scale(0.8);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }

    .image-container {
        text-align: center;
        margin: 20px 0;
    }

    .image-container img {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        max-width: 100%;
        height: auto;
        animation: fadeInZoom 2s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown(
    """
    <div class="image-container">
        <img src="https://i.postimg.cc/3RFFjwgN/regenerative-ag-image.avif" alt="Plant Disease Recognition">
    </div>
    """,
    unsafe_allow_html=True
)

# Divider
    st.markdown("---")

    


# --------------------------------------------------Features Section--------------------------------------------------------------------------------------

    st.markdown("<h2 style='text-align: center; color: {primary_green};'>🌟 Key Features 🌟</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="feature-card">
                <h3>🌾 AI-Powered Detection</h3>
                <p>State-of-the-art AI model trained to identify diseases in maize leaves with high accuracy.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="feature-card">
                <h3>📊 Detailed Reports</h3>
                <p>Get insights into the detected disease and recommendations for treatment.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="feature-card">
                <h3>🕛 Fast & Reliable</h3>
                <p>Quick predictions to save your valuable time and reduce crop losses.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("---")
    
# --------------------------------------------------END Features Section--------------------------------------------------------------------------------------


# -----------------------------------------------------How It Works Section------------------------------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: {primary_green};'>🚀 How It Works</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <ol class="home-content" style="text-align: left;">
            <li>Upload an image of a maize leaf through the <b>Disease Detection</b> page.</li>
            <li>Our AI model analyzes the image and predicts the disease.</li>
            <li>Receive actionable insights and recommendations instantly. </li>
        </ol>
        """, 
        unsafe_allow_html=True
    )

    st.markdown("---")
# -------------------------------------------------- END  Section--------------------------------------------------------------------------------------


#----------------------------------------------------- AI page start ---------------------------------------------------------------------------------
if app_mode == "🤖 AI Chat":

    #----------------------------------------------------- AI initial start ---------------------------------------------------------------------------------
    import streamlit as st
    import google.generativeai as genai

    # ————— One-time setup of model & session —————
    if "genai_session" not in st.session_state:
        genai.configure(api_key=st.secrets["google"]["api_key"])
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
        )
        st.session_state.genai_session = model.start_chat(
            history=[{
                "role": "user",
                "parts": [
                    "Waxaad tahay khabiir beeraleyda ah oo ku hadla af Soomaaliga. "
                    "Ka jawaab su'aalaha la xiriira cudurrada dhirta, bacriminta, waraabka, "
                    "iyo wax-soo-saarka beeraleyda si xirfad leh oo af Soomaali ah.",
                ],
            }]
        )

    chat_session = st.session_state.genai_session

    # ————— Display helper (unchanged) —————
    def display_message(sender, message, ua=None, ba=None):
        ua = ua or "https://example.com/avatar_user.png"
        ba = ba or "https://example.com/avatar_bot.png"
        if sender == "Adiga":
            _, tc, ac = st.columns([1, 8, 1])
            with tc:
                st.markdown(
                    f"<div style='text-align: right;'><strong>{sender}:</strong> {message}</div>",
                    unsafe_allow_html=True
                )
            with ac:
                st.image(ua, width=40)
        else:
            ac, tc, _ = st.columns([1, 8, 1])
            with ac:
                st.image(ba, width=40)
            with tc:
                st.markdown(f"**{sender}:** {message}")

# Ensure chat_history exists
    st.session_state.setdefault("chat_history", [])

    # Render all past messages
    for sender, msg in st.session_state.chat_history:
        display_message(sender, msg)

    # Callback that runs on form submit
    def handle_submit():
        prompt = st.session_state.user_prompt.strip()
        if not prompt:
            st.warning("Fadlan qor su'aal si aad u hesho jawaab.")
            return

        # 1) Echo user
        st.session_state.chat_history.append(("Adiga", prompt))

        # 2) Call AI inside a spinner
        with st.spinner("Agribot waxa uu ka shaqeynayaa…"):
            response = chat_session.send_message(prompt)
        ai_reply = response.text

        # 3) Append bot reply
        st.session_state.chat_history.append(("AgriBot", ai_reply))

    # Render the form with automatic clear on submit
    with st.form(key="chat_form", clear_on_submit=True):
        st.text_area(
            "Su’aashaada ku qor halkan (Af Soomaali):",
            key="user_prompt",
            height=100
        )
        st.form_submit_button("Dir", on_click=handle_submit)

    # 4) ————— NEW CHAT LOGIC SNIPPET ENDS HERE —————


# OLD CODE
    # def add_message(sender, message):
    #     st.session_state.chat_history.append((sender, message))
    

    # def display_message(sender, message, user_avatar=None, bot_avatar=None):
    #     """
    #     AI replies on the left (avatar + text),
    #     user messages on the right (text + avatar).
    #     """
    #     ua = user_avatar or "https://example.com/avatar_user.png"
    #     ba = bot_avatar  or "https://example.com/avatar_bot.png"
        
    #     if sender == "Adiga":  # user message
    #         # [spacer][ text ][ avatar ]
    #         spacer, text_col, avatar_col = st.columns([1, 8, 1])
    #         with text_col:
    #             # align the text to the right edge
    #             st.markdown(
    #                 f"<div style='text-align: right;'><strong>{sender}:</strong> {message}</div>",
    #                 unsafe_allow_html=True
    #             )
    #         with avatar_col:
    #             st.image(ua, width=40)
    #     else:  # AgriBot reply
    #         # [ avatar ][ text ][ spacer ]
    #         avatar_col, text_col, spacer = st.columns([1, 8, 1])
    #         with avatar_col:
    #             st.image(ba, width=40)
    #         with text_col:
    #             st.markdown(f"**{sender}:** {message}")


    
    # # 1Initialize chat history
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []

    # # Display existing chat history
    # for sender, msg in st.session_state.chat_history:
    #     display_message(
    #         sender,
    #         msg,
    #         user_avatar="https://example.com/avatar_user.png",
    #         bot_avatar="https://example.com/avatar_bot.png"
    #     )



    # st.markdown("### 🤖 La hadal Khabiirka Beeraha (AI)")
    # # User input and response
    # user_prompt = st.text_area("Su'aashaada ku qor halkan (Af Soomaali):", height=100)

    # if st.button("Dir"):
    #     if user_prompt.strip():
    #         # Add user message
    #         st.session_state.chat_history.append(("Adiga", user_prompt))
    #         # add_message("Adiga", user_prompt)

    #         # Get AI response
    #         response = chat_session.send_message(user_prompt)
    #         ai_reply = response.text

    #         # Add AI message
    #         st.session_state.chat_history.append(("AgriBot", ai_reply))
    #         # add_message("AgriBot", ai_reply)

    #         # Optional: play Somali speech
    #         #speak_somali(ai_reply)
    #     else:
    #         st.warning("Fadlan qor su'aal si aad u hesho jawaab.")


    
    
    
    # user_prompt = st.text_area("Su'aashaada ku qor halkan (Af Soomaali):", height=100)

    # if st.button("Dir"):
    #     if user_prompt.strip():
    #         response = chat_session.send_message(user_prompt)
    #         st.markdown("**Jawaabta Khabiirka:**")
    #         st.success(response.text)
    #     else:
    #         st.warning("Fadlan qor su'aal si aad u hesho jawaab.")





#----------------------------------------------------- AI page end ---------------------------------------------------------------------------------



#------------------------------------------------------------------- About page----------------------------------------------------------------------------


if app_mode == "📖 About":
    st.header("ℹ️ Plant Disease Recognition Model System")

    # Add description about the system
    st.markdown("""
    This **Model** leverages deep learning techniques to identify and diagnose plant diseases based on images of plant leaves.
    The system uses a **Convolutional Neural Network (CNN)** architecture, which has been trained on a large dataset of plant images, allowing it to accurately detect and classify a wide variety of diseases. 
    The model is continuously updated with new data to improve its predictions. This system is designed to be used by Government, farmers, researchers, and plant enthusiasts to quickly detect diseases in their crops and take timely action to mitigate potential crop damage.
    
                
    ### How It Works
    1. **Image Upload:** Users upload an image of a plant leaf affected by a potential disease.
    2. **Model Prediction:** The system uses a pre-trained deep learning model to analyze the uploaded image and predict the disease.
    3. **Disease Classification:** The system classifies the image into one of the pre-defined categories, such as healthy or affected by a specific disease.
    4. **Diagnosis Report:** After the prediction, a detailed report is generated with the disease name and management suggestions.
                

    ### About Dataset
    This dataset has been made using the popular PlantVillage and PlantDoc datasets. During the formation of the dataset certain images have been removed which were not found to be useful. The original authors reserve right to the respective datasets.
     If you use this dataset in your academic research, please credit the authors. 
                
    **Dataset Description**: A dataset for classification of corn or maize plant leaf diseases
                
    1: Common Rust - 1306 images\n\n 
    2: Gray Leaf Spot - 574 images\n\n
    3: Blight -1146 images\n\n 
    4: Healthy - 1162 images

    
    #### Citations:

    Singh D, Jain N, Jain P, Kayal P, Kumawat S, Batra N. PlantDoc: a dataset for visual plant disease detection. InProceedings of the 7th ACM IKDD CoDS and 25th COMAD 2020 Jan 5 (pp. 249-253).

    J, ARUN PANDIAN; GOPAL, GEETHARAMANI (2019), “Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network”, Mendeley Data, V1, doi: 10.17632/tywbtsjrjv.1


    ### Benefits of Using the System
    - **Fast Results:** The system provides instant disease predictions, allowing users to act quickly.
    - **Wide Range of Diseases:** The model supports multiple plant diseases, covering a wide range of crops and fruits.
    - **Accuracy:** The system is trained to deliver accurate results based on large datasets of plant images, ensuring reliable predictions.
    - **Accessible and User-friendly:** No prior technical knowledge is required. Simply upload an image, and the system will take care of the rest.

    ### Supported Plants
    Currently our **MVP model system** support one plant leaf:
    - Maize 🌽



    """)

    # ------------------------------------------------------------Styling for the About Page to match the theme---------------------------------------------
    st.markdown(
    f"""
    <style>
    .about-header {{
        text-align: center;
        color: {primary_green};
        font-size: 2.5em;
        margin-bottom: 30px;
    }}
    .about-content {{
        text-align: center;
        font-size: 1.2em;
        line-height: 1.6;
        color: {text_gray};
        margin-bottom: 40px;
    }}
    .about-content a {{
        color: {primary_green};
        text-decoration: none;
    }}
    .about-content a:hover {{
        color: {accent_yellow};
        text-decoration: underline;
    }}
    .section-title {{
        font-size: 2em;
        font-weight: bold;
        color: {primary_green};
        margin-top: 40px;
        text-align: center;
    }}
    .team-card {{
        background: linear-gradient(135deg, #3726a6, #06d063);
        padding: 20px;
        border-radius: 10px;
        margin: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 350px; /* Set a fixed height for all cards */
    }}
    .team-card img {{
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 15px;
    }}
    .team-card:hover {{
        transform: scale(1.05);
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.2);
    }}
    .team-card h3 {{
        color: {dark_olive_green};
        font-size: 1.8em;
        margin-bottom: 10px;
    }}
    .team-card p {{
        color: {text_gray};
        font-size: 1.2em;
    }}
    .footer {{
        text-align: center;
        padding: 25px;
        background-color: {soft_beige};
        margin-top: 40px;
        border-radius: 20px;
    }}
    .footer p {{
        font-size: 1.2em;
        color: {primary_green};
    }}

    /* Responsive Design */
    @media (max-width: 1200px) {{
        .about-header {{
            font-size: 2.2em;
        }}
        .section-title {{
            font-size: 1.8em;
        }}
        .team-card {{
            padding: 15px;
            margin: 10px;
        }}
        .footer p {{
            font-size: 1.1em;
        }}
    }}

    @media (max-width: 768px) {{
        .about-header {{
            font-size: 2em;
        }}
        .section-title {{
            font-size: 1.6em;
        }}
        .team-card {{
            padding: 10px;
            margin: 8px;
        }}
        .about-content {{
            font-size: 1.1em;
        }}
        .footer p {{
            font-size: 1em;
        }}
        .team-card h3 {{
            font-size: 1.5em;
        }}
        .team-card p {{
            font-size: 1.1em;
        }}
    }}

    @media (max-width: 480px) {{
        .about-header {{
            font-size: 1.8em;
        }}
        .section-title {{
            font-size: 1.5em;
        }}
        .team-card {{
            padding: 8px;
            margin: 5px;
        }}
        .about-content {{
            font-size: 1em;
        }}
        .footer p {{
            font-size: 0.9em;
        }}
        .team-card h3 {{
            font-size: 1.3em;
        }}
        .team-card p {{
            font-size: 1em;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)



    # ---------------------------------------------Ending with a footer or call-to-action for the app----------------------------------------------------------
    st.markdown(f"""
    <div class="footer">
        <p>Explore more and start detecting plant diseases today with the Plant Disease Recognition System! 🌿</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------Ending with a footer or call-to-action for the app---------------------------------------------------









# --------------------------------------------Disease Detection Page------------------------------------------------------------------
if app_mode == "🌿 Disease Detection":
    st.header("🌿 Disease Detection")
    st.write("Upload an image of a maize leaf to detect potential diseases 📸")

    # -----------------------------------------------------File uploader for image--------------------------------------------------------

    # ----------------------------------- #halkan waa halkan aan badalka sameeyey.
    # Somali: Labada ikhtiyaar ee sawirka — Upload ama Kamarad; mid kasta wuu shaqaynayaa
    tab_upload, tab_camera = st.tabs(["📤 Upload", "📷 Camera"])  # Somali: Tabs si beeralaydu u doortaan

    with tab_upload:
        # Somali: Isla uploader-kii hore oo aan beddelin fahamka koodhka intiisa kale
        test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    with tab_camera:
        # Somali: Kamaradda mobilka/PC-ga toos uga qaad sawirka caleenta
        camera_photo = st.camera_input("Sawir ka qaad caleenta (kamarad)")

    # Somali: Haddii kamarad la adeegsaday, dooro taas; haddii kale ka qaado upload
    img_for_model = camera_photo or test_image

    if img_for_model is not None:
        # Somali: Tus sawirka la doortay si degdeg ah loo xaqiijiyo
        st.image(img_for_model, caption="Uploaded Image", width=300, use_column_width=False)  # Adjusted width

        # ---------------------------------------------------------Predict button--------------------------------------------------------------
        if st.button("Predict"):
            with st.spinner("Analyzing the image...............................................................................!"):
                try:
                    # Model Prediction (Somali: Halkaan waxaan u gudbineynaa img_for_model si uploader/camera labaduba u shaqeeyaan)
                    result_index, confidence_pct = model_prediction(img_for_model)  # modify
    # halkan waa halkaan ku joogeye.

                    # Labels
                    class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']
                    disease_name = class_names[result_index]
                    #confidence_pct = round(confidence * 100, 2)
                    
                    #st.write(f"*Analyzing the image...*: {disease_name}")
                    #st.write(f"*Confidence Level*: {confidence_pct}%")

                    if disease_name != "Healthy":
                        st.warning(f"The leaf is affected by *{disease_name}* ⚠")
                        # Model Performance Section (inside Disease Detection page)
                        st.markdown("<h2 style='text-align: center; color: {primary_green};'>📊 Model Performance</h2>", unsafe_allow_html=True)
                        container = st.container()
                        container.write(f"The disease found on this leaf is *{disease_name}*, with a confidence level of {confidence_pct}%")
                                  
                        if disease_name == "Blight":
                            st.warning("""
                            Xanuunka laga helay caleentan waxa loo yaqaan qoyaan-caaryo burbur (Northern corn leaf blight)          
                            Northern corn leaf blight (NCLB) in maize, caused by the fungus Exserohilum turcicum, is a significant concern in regions like Somaliland. 
                            This disease is characterized by long, cigar-shaped lesions on the leaves, which can coalesce and cause extensive leaf damage, leading to substantial yield losses. 
                            NCLB thrives in warm, humid conditions, which are common in Somaliland. 
                            
                            ### Appropriate Possible Solutions (Suggestion)          
                            Quick control measures include planting resistant maize varieties, practicing crop rotation to reduce fungal spores in the soil, and applying fungicides such as Azoxystrobin or Propiconazole at the early stages of infection. 
                            Additionally, proper field sanitation, including the removal of infected plant debris, and timely monitoring of crops for early signs of the disease are crucial steps to manage and mitigate the impact of this disease.      
                           
                            *If you need further details or guidance on managing this disease, please contact agriculture consultant as quick as possible*
                                        """)
                            
                        elif disease_name == "Common Rust":
                            st.warning("""
                            Xanuunka laga helay caleentan waxa loo yaqaan caabuq daxaleed (Common Rust) 
                            Common Rust in maize, caused by the fungus Puccinia sorghi, is a prevalent issue in Somaliland. 
                            This disease is identified by small, circular, cinnamon-brown pustules on both sides of the leaves, which can darken as the plant matures. 
                            The disease thrives in cool, moist conditions, typically between 15-25°C, and high humidity. Infected plants may exhibit chlorosis (yellowing) and premature leaf death, leading to significant yield losses. 
                            
                            ### Appropriate Possible Solutions (Suggestion)          
                            Quick control measures include planting resistant maize varieties, applying foliar fungicides such as mancozeb, pyraclostrobin, or azoxystrobin + propiconazole early in the season, and practicing crop rotation to reduce the presence of the fungus in the soil. 
                            Additionally, removing and destroying infected plant debris and monitoring crops regularly for early signs of infection are crucial steps to manage and mitigate the impact of this disease.
                            
                            *If you need further details or guidance on managing this disease, please contact agriculture consultant as quick as possible*
                                        """)
                            
                        elif disease_name == "Gray Leaf Spot":
                            st.warning("""
                            Xanuunka laga helay caleentan waxa loo yaqaan bal-bal bareed (Gray Leaf Spot) 
                            Gray Leaf Spot (GLS) in maize, caused by the fungus Cercospora zeae-maydis, is a significant threat to maize production in Somaliland. 
                            This disease is characterized by small, rectangular, brown to gray lesions that run parallel to the leaf veins. 
                            These lesions can coalesce, leading to extensive leaf blight and significant yield losses. GLS thrives in warm, humid conditions, which are common in Somaliland. 
                            The fungus survives in crop residue and spreads through wind and rain splash.
                            
                                       
                            ### Appropriate Possible Solutions (Suggestion)          
                            Quick control measures include planting resistant maize varieties, practicing crop rotation to reduce the presence of the fungus in the soil, and incorporating crop residues into the soil through tillage to promote decomposition. 
                            Applying fungicides such as strobilurins (e.g., Azoxystrobin) or triazoles (e.g., Propiconazole) at the early stages of infection can also be effective. Additionally, ensuring proper field sanitation by removing and destroying infected plant debris and regularly 
                            monitoring crops for early signs of infection are crucial steps to manage and mitigate the impact of this disease.
                            
                            *If you need further details or guidance on managing this disease, please contact agriculture consultant as quick as possible*
                                        """)

                    else:
                        st.snow()
                        st.success(f"The leaf is *{disease_name}*. No diseases detected. ✅")

                        st.success("""
                            Congradulation! your plant is healthy. To ensure your maize leaves remain healthy and productive, regularly monitor your plants for any signs of disease or pest damage, as early detection is crucial. 
                            Use balanced fertilization to provide essential nutrients like nitrogen, phosphorus, and potassium, which are vital for leaf development. Ensure adequate irrigation, especially during critical growth stages, but avoid overwatering to prevent root diseases. 
                            Practice crop rotation to reduce soil-borne pathogens and pests, and plant resistant maize varieties to minimize the risk of infection. Maintain proper field sanitation by removing and destroying plant debris from previous crops to prevent pathogen overwintering. 
                            Implement integrated pest management (IPM) strategies to control pests effectively, and apply fungicides promptly if early signs of fungal diseases are detected. By following these suggestions, you can help keep your maize leaves healthy and your crops productive. 
                            
                            *If you need more detailed advice on any of these points, feel free to contact agriculture consultant!.
                                        """)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.warning("""Please upload an image of leaf to proceed. 
                   Sawir maad so gudbin weli, fadlan soo gudbi sawir ka caleenta geedka ⚠
                   
                   """)



#  -------------------------------------------------------END Predict button--------------------------------------------------------------------------------------------       



# # --------------------------------------------Disease Detection Page Old one (there is no camera in this)------------------------------------------------------------------
# if app_mode == "🌿 Disease Detection":
#     st.header("🌿 Disease Detection")
#     st.write("Upload an image of a maize leaf to detect potential diseases 📸")

#     # -----------------------------------------------------File uploader for image--------------------------------------------------------

#     test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

#     if test_image is not None:
#         st.image(test_image, caption="Uploaded Image", width=300, use_column_width=False)  # Adjusted width

#         # ---------------------------------------------------------Predict button--------------------------------------------------------------
#         if st.button("Predict"):
#             with st.spinner("Analyzing the image...............................................................................!"):
#                 try:
#                     # Model Prediction
#                     result_index, confidence_pct = model_prediction(test_image) # modify

#                     # Labels
#                     class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']
#                     disease_name = class_names[result_index]
#                     #confidence_pct = round(confidence * 100, 2)
                    
#                     #st.write(f"**Analyzing the image...**: {disease_name}")
#                     #st.write(f"**Confidence Level**: {confidence_pct}%")

#                     if disease_name != "Healthy":
#                         st.warning(f"The leaf is affected by **{disease_name}** ⚠️")
#                         # Model Performance Section (inside Disease Detection page)
#                         st.markdown("<h2 style='text-align: center; color: {primary_green};'>📊 Model Performance</h2>", unsafe_allow_html=True)
#                         container = st.container()
#                         container.write(f"The disease found on this leaf is **{disease_name}**, with a confidence level of {confidence_pct}%")
                                  
#                         if disease_name == "Blight":
#                             st.warning("""
#                             Xanuunka laga helay caleentan waxa loo yaqaan qoyaan-caaryo burbur (Northern corn leaf blight)          
#                             Northern corn leaf blight (NCLB) in maize, caused by the fungus Exserohilum turcicum, is a significant concern in regions like Somaliland. 
#                             This disease is characterized by long, cigar-shaped lesions on the leaves, which can coalesce and cause extensive leaf damage, leading to substantial yield losses. 
#                             NCLB thrives in warm, humid conditions, which are common in Somaliland. 
                            
#                             ### Appropriate Possible Solutions (Suggestion)          
#                             Quick control measures include planting resistant maize varieties, practicing crop rotation to reduce fungal spores in the soil, and applying fungicides such as Azoxystrobin or Propiconazole at the early stages of infection. 
#                             Additionally, proper field sanitation, including the removal of infected plant debris, and timely monitoring of crops for early signs of the disease are crucial steps to manage and mitigate the impact of this disease.      
                           
#                             **If you need further details or guidance on managing this disease, please contact agriculture consultant as quick as possible**
#                                         """)
                            
#                         elif disease_name == "Common Rust":
#                             st.warning("""
#                             Xanuunka laga helay caleentan waxa loo yaqaan caabuq daxaleed (Common Rust) 
#                             Common Rust in maize, caused by the fungus Puccinia sorghi, is a prevalent issue in Somaliland. 
#                             This disease is identified by small, circular, cinnamon-brown pustules on both sides of the leaves, which can darken as the plant matures. 
#                             The disease thrives in cool, moist conditions, typically between 15-25°C, and high humidity. Infected plants may exhibit chlorosis (yellowing) and premature leaf death, leading to significant yield losses. 
                            
#                             ### Appropriate Possible Solutions (Suggestion)          
#                             Quick control measures include planting resistant maize varieties, applying foliar fungicides such as mancozeb, pyraclostrobin, or azoxystrobin + propiconazole early in the season, and practicing crop rotation to reduce the presence of the fungus in the soil. 
#                             Additionally, removing and destroying infected plant debris and monitoring crops regularly for early signs of infection are crucial steps to manage and mitigate the impact of this disease.
                            
#                             **If you need further details or guidance on managing this disease, please contact agriculture consultant as quick as possible**
#                                         """)
                            
#                         elif disease_name == "Gray Leaf Spot":
#                             st.warning("""
#                             Xanuunka laga helay caleentan waxa loo yaqaan bal-bal bareed (Gray Leaf Spot) 
#                             Gray Leaf Spot (GLS) in maize, caused by the fungus Cercospora zeae-maydis, is a significant threat to maize production in Somaliland. 
#                             This disease is characterized by small, rectangular, brown to gray lesions that run parallel to the leaf veins. 
#                             These lesions can coalesce, leading to extensive leaf blight and significant yield losses. GLS thrives in warm, humid conditions, which are common in Somaliland. 
#                             The fungus survives in crop residue and spreads through wind and rain splash.
                            
                                       
#                             ### Appropriate Possible Solutions (Suggestion)          
#                             Quick control measures include planting resistant maize varieties, practicing crop rotation to reduce the presence of the fungus in the soil, and incorporating crop residues into the soil through tillage to promote decomposition. 
#                             Applying fungicides such as strobilurins (e.g., Azoxystrobin) or triazoles (e.g., Propiconazole) at the early stages of infection can also be effective. Additionally, ensuring proper field sanitation by removing and destroying infected plant debris and regularly 
#                             monitoring crops for early signs of infection are crucial steps to manage and mitigate the impact of this disease.
                            
#                             **If you need further details or guidance on managing this disease, please contact agriculture consultant as quick as possible**
#                                         """)

#                     else:
#                         st.snow()
#                         st.success(f"The leaf is **{disease_name}**. No diseases detected. ✅")

#                         st.success("""
#                             Congradulation! your plant is healthy. To ensure your maize leaves remain healthy and productive, regularly monitor your plants for any signs of disease or pest damage, as early detection is crucial. 
#                             Use balanced fertilization to provide essential nutrients like nitrogen, phosphorus, and potassium, which are vital for leaf development. Ensure adequate irrigation, especially during critical growth stages, but avoid overwatering to prevent root diseases. 
#                             Practice crop rotation to reduce soil-borne pathogens and pests, and plant resistant maize varieties to minimize the risk of infection. Maintain proper field sanitation by removing and destroying plant debris from previous crops to prevent pathogen overwintering. 
#                             Implement integrated pest management (IPM) strategies to control pests effectively, and apply fungicides promptly if early signs of fungal diseases are detected. By following these suggestions, you can help keep your maize leaves healthy and your crops productive. 
                            
#                             **If you need more detailed advice on any of these points, feel free to contact agriculture consultant!.*
#                                         """)

#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")
#     else:
#         st.warning("""Please upload an image of leaf to proceed. 
#                    Sawir maad so gudbin weli, fadlan soo gudbi sawir ka caleenta geedka ⚠️
                   
#                    """)



# #  -------------------------------------------------------END Predict button--------------------------------------------------------------------------------------------       


# ----------------------------------------------------Adding Custom CSS for Responsiveness---------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stat-card {{
        background: linear-gradient(135deg, #3726a6, #06d063);
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
    }}
    .stat-card h3 {{
        font-size: 1.2em;
        color: {primary_green};
    }}
    .feature-card {{
        background-color: {secondary_light_green};
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        text-align: center;
    }}
    .feature-card h3 {{
        color: {dark_olive_green};
        font-size: 1.5em;
    }}
    .btn {{
        background-color: {accent_yellow};
        color: {dark_olive_green};
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }}
    .btn:hover {{
        background-color: {warm_brown};
    }}
    .main-content {{
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    /* Mobile and small screen responsiveness */
    @media (max-width: 768px) {{
        .stat-card, .feature-card {{
            width: 100%;
        }}
        .feature-card h3 {{
            font-size: 1.2em;
        }}
        .stat-card h3 {{
            font-size: 1em;
        }}
    }}
    </style>
    """, 
    unsafe_allow_html=True
)


# ------------------------------------------------------------------------------------------------------------------------\




