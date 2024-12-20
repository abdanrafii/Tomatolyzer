# Google Drive file IDs
MODEL_FILE_ID = "1NicVqNqoQewx0FykWd5T92a0ufkys2lC"  # Replace with the FILE_ID of PlantTomatoDisease.h5
CLASS_NAMES_FILE_ID = "1AXBXtJhHtvU_oUDOISrxiHbjVEZ2BIPA"  # Replace with the FILE_ID of class_names.json


# Paths to save the files locally
MODEL_PATH = "PlantTomatoDisease.h5"
CLASS_NAMES_PATH = "class_names.json"

# Download model file
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id=1NicVqNqoQewx0FykWd5T92a0ufkys2lC", MODEL_PATH, quiet=False)

# Download class names file
if not os.path.exists(CLASS_NAMES_PATH):
    gdown.download(f"https://drive.google.com/uc?id=1AXBXtJhHtvU_oUDOISrxiHbjVEZ2BIPA", CLASS_NAMES_PATH, quiet=False)

api_key = st.secrets["api_keys"]["API_KEY"]

client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="Tomatolyzer",
    page_icon="🍅",
    layout="wide"
)

# Apply custom CSS for card design
st.markdown(
    """
    <style>
    .card {
        box-shadow: 0 4px 8px 0 rgba(128, 128, 128, 0.3);
        transition: 0.3s;
        padding: 16px;
        margin: 16px 0;
        border-radius: 8px;
    }
    .card-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .card-content {
        font-size: 16px;
        margin-bottom: 12px;
    }
    .stTextInput {
        margin-top: -27px;
    }
    .alert {
    padding: 16px;
    border-radius: 5px;
    margin-bottom: 15px;
    }
    .success { background-color: #d4edda; color: #155724; }
    .info { background-color: #cce5ff; color: #004085; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Streamlit app
st.title("🍅 Tomatolyzer")
st.markdown("### A Tomato Plant Disease Classification App")

# Load model
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# # Load model
# @st.cache_resource
# def load_trained_model():
#     model_path = "/content/PlantTomatoDisease.h5"  # Update this with your model's saved path
#     return load_model(model_path)

# model = load_trained_model()

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

# # Load class names
# with open('class_names.json', 'r') as f:
#     class_names = json.load(f)

# menu = st.sidebar.selectbox(
#     "Navigasi Menu",
#     ["Home", "Upload Gambar", "Tanya - Jawab Interaktif"],
#     index=0,
# )

tab1, tab2, tab3 = st.tabs(["Home", "Analyze", "Chatbot"])

# st.sidebar.markdown('[Link GitHub](https://github.com/abdanrafii/Final-Project-Startup-Campus-Tim-1)')

with tab1:
  # st.header("Home")
  # st.write("Tomatolyzer")
  st.write("""
  **Penyakit yang dapat dideteksi:**
  - Tomato Bacterial Spot
  - Tomato Early Blight
  - Tomato Late Blight
  - Tomato Leaf Mold
  - Tomato Septoria Leaf Spot
  - Tomato Spider Mites
  - Tomato Target Spot
  - Tomato Mosaic Virus
  """)
with tab2:
  # st.header("Upload Gambar")
  st.write("""
  **Petunjuk Penggunaan:**
  1. Unggah Gambar
  2. Tunggu Proses Analisis
  3. Lihat Hasil Deteksi
  """)
  # # Sample images
  # sample_images_folder = 'sample_images'
  # sample_images = [f for f in os.listdir(sample_images_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

  # # Option to choose between uploaded or sample image
  # image_choice = st.selectbox("Pilih opsi gambar:", ["Unggah gambar sendiri", "Pilih gambar dari sampel"])

  # File uploader
  # if image_choice == "Unggah gambar sendiri":
  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
  if uploaded_file is not None:
          selected_image = uploaded_file
          image_caption = "Uploaded Image"
          image_name = uploaded_file.name
  else:
          selected_image = None
          image_caption = "No uploaded image"

  if selected_image:
      # Handle image opening based on its type
      if isinstance(selected_image, str):
          img = Image.open(selected_image)
      else:
          img = Image.open(selected_image)

      # Convert image to base64 and preprocess
      buffered = io.BytesIO()
      img.save(buffered, format="PNG")
      img_base64 = base64.b64encode(buffered.getvalue()).decode()

      # Preprocess the image
      img = img.resize((254, 254))
      img = img.convert("RGB")
      img_array = image.img_to_array(img)
      img_array = np.expand_dims(img_array, axis=0)

      # Predict the class
      with st.spinner("Classifying..."):
          predictions = model.predict(img_array, verbose=0)
          predicted_class = class_names[np.argmax(predictions, axis=1)[0]]
          confidence = np.max(predictions) * 100

      st.markdown(f"""
      <div class="card">
          <div class="card-title">Uploaded Image</div>
          <div class="card-content">Image uploaded: <b>{image_name}</b></div>
          <div class="card-content">
          <img src="data:image/png;base64,{img_base64}" style="width:100%; border-radius: 8px">
          <div class="alert success" style="margin-top: 20px;">Prediction: {predicted_class}</div>
          <div class="alert info">Confidence: {confidence:.2f}%</div>
      </div>
      """, unsafe_allow_html=True)

      if predicted_class == "Tomato healthy":
          chat_content = (
              "You are a plant pathologist. The tomato plant appears to be healthy. Please provide short, positive and informative message about the general care and maintenance of healthy tomato plants, including:\n\n"
              "1. General Care: Tips for keeping tomato plants healthy, including watering, sunlight, and nutrition.\n"
              "2. Prevention: Steps to prevent common diseases and pestsr.\n"
              "3. Harvesting: When and how to harvest tomatoes for the best quality.\n"
              "4. Future Planning: Advice on crop rotation and maintaining soil health for future plantings.\n\n"
              "\nYou must answer in indonesian language. Make the answer short"
          )
      else:
          chat_content = (
              f"You are a plant pathologist. Provide a brief yet informative analysis of {predicted_class} that includes:"
              "\nDisease Overview & Symptoms: Summarize the disease and its main symptoms in one or two sentences."
              "\nCause & Spread: Explain the pathogen and its transmission in a simple sentence."
              "\nImmediate Treatment Actions: List one or two key actions to take upon detection."
              "\nPrevention Tips: Share a quick prevention tip."
              "\nLong-term Management: Suggest one long-term strategy for prevention or control."
              "\n\nYou must answer in indonesian language"
          )

      # Function to get disease information using OpenAI
      chat_completion = client.chat.completions.create(
      messages = [
              {
                  "role": "user",
                  "content": chat_content
              }
          ],
          model="gpt-3.5-turbo",  # Ensure this model is available in your account
      )

      # Get disease information from OpenAI
      with st.spinner("Fetching additional information..."):
          chatgpt_response = chat_completion.choices[0].message.content.strip()

      # Card 2
      st.markdown(
          f"""
          <div class="card">
              <div class="card-title">Berikut merupakan Informasi yang mungkin dapat membantu</div>
              <div class="card-content">
                  {chatgpt_response}
              </div>
          </div>
          """,
          unsafe_allow_html=True,
      )
with tab3:

  # st.write("""
  # **Petunjuk Penggunaan:**
  # 1. Ketik pertanyaan yang ingin ditulis dan klik enter terlebih dahulu
  # 2. Klik tombol kirim dan tunggu beberapa saat
  # 3. Maka, hasil dari pertanyaan tersebut akan muncul
  # """)
  # st.header("Tanya - Jawab Interaktif")
  def is_question_in_context(question):
    context_check_prompt = (
        "Apakah pertanyaan berikut terkait dengan 'penyakit pada tanaman tomat'? Jawab dengan 'Ya' atau 'Tidak'.\n\n"
        f"Pertanyaan: {question}"
    )

    # Correct API call for non-chat model
    response = client.completions.create(
        prompt=context_check_prompt,
        model="gpt-3.5-turbo-instruct"  # Model suitable for completions endpoint
    )
    answer = response.choices[0].text.strip().lower()
    # st.info(response)
    # st.info("Testing (abaikan): " + answer)
    return answer == "ya"


    # Fungsi untuk mendapatkan jawaban dari OpenAI
  def get_answer(question):
      response = client.chat.completions.create(
          messages = [
              {
                  "role": "user",
                  "content": f"Pertanyaan: {question}\nJawaban mengenai penyakit tanaman tomat:"
              }
          ],

          model="gpt-3.5-turbo",
      )
      return response.choices[0].message.content.strip()

    # Input pengguna
  user_question = st.text_input("", placeholder="Masukkan pertanyaan tentang penyakit tomat Anda di sini")

  if st.button("Kirim"):
      if user_question:
          if is_question_in_context(user_question):
              answer = get_answer(user_question)
              st.success(f"Jawaban: {answer}")
          else:
              st.warning("Pertanyaan ini di luar konteks 'penyakit pada tanaman tomat'. Silakan ajukan pertanyaan yang relevan.")
      else:
          st.warning("Silakan masukkan pertanyaan!")
