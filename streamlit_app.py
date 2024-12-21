import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import streamlit as st
import json
from PIL import Image
import io
import base64
import gdown
from openai import OpenAI

client = OpenAI(api_key=st.secrets["API_KEY"])

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
    
# Load model
# @st.cache_resource
# def load_trained_model():
#     return load_model(MODEL_PATH)

@st.cache_resource
def load_trained_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except ValueError as e:
        st.error(f"Failed to load the model: {e}")
        return None
        
model = load_trained_model()

# def get_model_summary(model):
#     stream = io.StringIO()  # Create a stream to capture the summary
#     model.summary(print_fn=lambda x: stream.write(x + "\n"))  # Redirect summary to the stream
#     summary_str = stream.getvalue()  # Get the content of the stream
#     stream.close()  # Close the stream
#     return summary_str

# # Display the model summary in Streamlit
# st.title("Model Architecture")
# st.text(get_model_summary(model))

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
    
menu = st.sidebar.selectbox(
    "Navigasi Menu",
    ["Home", "Upload Gambar", "Tanya - Jawab Interaktif"],
    index=0,
)

# def get_readable_size(size_in_bytes):
#     """Convert bytes to a human-readable format."""
#     for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
#         if size_in_bytes < 1024:
#             return f"{size_in_bytes:.2f} {unit}"
#         size_in_bytes /= 1024
#     return f"{size_in_bytes:.2f} PB"

# def list_files_with_sizes(directory):
#     """List files in the directory with their sizes."""
#     try:
#         # Get all files and directories
#         with os.scandir(directory) as entries:
#             st.info(f"{'File Name':<40} {'Size':>10}")
#             st.info("-" * 50)
#             for entry in entries:
#                 if entry.is_file():  # Check if it's a file
#                     size = os.path.getsize(entry.path)
#                     st.info(f"{entry.name:<40} {get_readable_size(size):>10}")
#     except Exception as e:
#         st.info(f"Error: {e}")

# # Replace 'your_directory_path' with the path of the directory you want to scan
# directory_path = '/mount/src/tomatolyzer/'
# list_files_with_sizes(directory_path)

if menu == "Home":
  # st.header("Home")
  # st.write("Tomatolyzer")
  st.write("""
  #### Selamat Datang di Tomatolyzer!
  Tomatolyzer adalah platform berbasis Artificial Intelligence yang dirancang untuk mendeteksi penyakit pada tanaman tomat melalui analisis gambar daun. Dengan teknologi canggih yang menggabungkan Computer Vision dan Machine Learning, aplikasi ini memberikan solusi cepat, akurat, dan praktis bagi petani maupun pelaku agribisnis dalam menjaga kesehatan tanaman mereka.

  #### Fitur Utama:
  ###### 1. Deteksi Penyakit Otomatis
  Unggah gambar daun tomat pada tab "Upload Gambar", dan sistem akan menganalisis secara otomatis dengan tingkat akurasi tinggi.

  ###### 2. Informasi dan Solusi Penyakit
  Setelah deteksi, pengguna akan menerima informasi lengkap mulai dari:
  - Nama penyakit
  - Gejala umum
  - Penyebab utama
  - Langkah penanganan
  - Saran jangka panjang untuk mencegah penyakit berulang

 ###### 3. Tanya - Jawab Interaktif
 Fitur ini memanfaatkan AI dari OpenAI yang memungkinkan pengguna untuk mengajukan pertanyaan terkait kesehatan tanaman tomat dan mendapatkan jawaban real-time.

 #### Teknologi di Balik Tomatolyzer:
 ###### Model Machine Learning:
 Menggunakan EfficientNetB3 yang terkenal karena efisiensinya dalam menangani variasi gambar dan meminimalkan kebutuhan daya komputasi, sambil tetap mempertahankan akurasi yang tinggi. Model ini telah dilatih dengan dataset spesifik tanaman tomat, mencakup 9 kelas penyakit utama.

 ###### Proses Deteksi Cepat:
 Gambar yang diunggah akan diproses dalam hitungan detik melalui pipeline deteksi yang mencakup preprocessing, ekstraksi fitur, dan prediksi akhir dengan akurasi hingga 99%.

 #### Penyakit yang dapat dideteksi:
  - Tomato Bacterial Spot
  - Tomato Early Blight
  - Tomato Late Blight
  - Tomato Leaf Mold
  - Tomato Septoria Leaf Spot
  - Tomato Spider Mites
  - Tomato Target Spot
  - Tomato Mosaic Virus

 #### Cara Penggunaan:
 ###### 1. Pada Deteksi Penyakit
 - Pindah ke tab "Upload Gambar"
 - Siapkan gambar daun tomat yang ingin diuji dan unggah gambar atau pilih salah satu sampel gambar
 - Tunggu beberapa saat dan lihat hasil deteksi
 - Maka, hasil penyakit akan keluar dan terdapat informasi lengkap mulai dari Nama Penyakit hingga Saran kedepannya

 ###### 2. Pada Tanya - Jawab Interaktif
 - Pindah ke tab "Tanya - Jawab Interaktif"
 - Ketik pertanyaan yang ingin ditulis dan klik enter terlebih dahulu\
 - Klik tombol kirim dan tunggu beberapa saat
 - Maka, hasil dari pertanyaan tersebut akan muncul

 ### Tomatolyzer hadir untuk memastikan tanaman tomat Anda tetap sehat dan produktif! 🌱🍅
  """)
elif menu == "Upload Gambar":
  # st.header("Upload Gambar")
  st.write("""
  **Petunjuk Penggunaan:**
  1. Siapkan gambar daun tomat yang ingin diuji dan unggah gambar atau pilih salah satu sampel gambar
  2. Tunggu beberapa saat dan lihat hasil deteksi
  3. Maka, hasil penyakit akan keluar dan terdapat informasi lengkap mulai dari Nama Penyakit hingga Saran kedepannya
  """)
  # Sample images
  sample_images_folder = 'sample_images'
  sample_images = [f for f in os.listdir(sample_images_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

  # Option to choose between uploaded or sample image
  image_choice = st.selectbox("Pilih opsi gambar:", ["Unggah gambar sendiri", "Pilih gambar dari sampel"])

  # File uploader
  if image_choice == "Unggah gambar sendiri":
      uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
      if uploaded_file is not None:
          selected_image = uploaded_file
          image_caption = "Uploaded Image"
          image_name = uploaded_file.name
      else:
          selected_image = None
          image_caption = "No uploaded image"

  elif image_choice == "Pilih gambar dari sampel":
      selected_sample = st.selectbox("Pilih gambar:", ["None"] + sample_images)
      if selected_sample != "None":
          selected_image = os.path.join(sample_images_folder, selected_sample)
          image_caption = f"Sample Image: {selected_sample}"
          image_name = os.path.basename(selected_sample)
      else:
          selected_image = None
          image_caption = "No sample image selected"

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

      # Adjust batch dimension based on model's expected input shape
      if model.input_shape == (None, 254, 254, 3):  # Model expects batch dimension
          img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
      elif model.input_shape == (254, 254, 3):  # Model does NOT expect batch dimension
          img_array = np.squeeze(img_array, axis=0)  # Remove batch dimension if added earlier

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
elif menu == "Tanya - Jawab Interaktif":

  st.write("""
  **Petunjuk Penggunaan:**
  1. Ketik pertanyaan yang ingin ditulis dan klik enter terlebih dahulu
  2. Klik tombol kirim dan tunggu beberapa saat
  3. Maka, hasil dari pertanyaan tersebut akan muncul
  """)
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
  user_question = st.text_input("**Masukkan pertanyaan terkait penyakit tomat:**", key="custom-text-input")

  if st.button("Kirim"):
      if user_question:
          if is_question_in_context(user_question):
              answer = get_answer(user_question)
              st.success(f"Jawaban: {answer}")
          else:
              st.warning("Pertanyaan ini di luar konteks 'penyakit pada tanaman tomat'. Silakan ajukan pertanyaan yang relevan.")
      else:
          st.warning("Silakan masukkan pertanyaan!")
