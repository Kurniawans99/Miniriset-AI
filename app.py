import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import keras
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import math
import os
import gdown

# --- FUNGSI BARU UNTUK DOWNLOAD MODEL ---
def download_model_from_gdrive(file_id, output_path):
    """
    Mengecek jika model ada, jika tidak, unduh dari Google Drive.
    """
    if not os.path.exists(output_path):
        with st.spinner(f"Mengunduh model besar ({output_path})... Ini mungkin butuh beberapa menit."):
            try:
                gdown.download(id=file_id, output=output_path, quiet=False)
                st.success(f"Model {output_path} berhasil diunduh.")
            except Exception as e:
                st.error(f"Gagal mengunduh model. Error: {e}")
                st.stop() # Hentikan eksekusi aplikasi jika model gagal diunduh

# --- PANGGIL FUNGSI DOWNLOAD SEBELUM MEMUAT MODEL ---
# Ganti ID di bawah ini dengan ID file Google Drive Anda
ALEXNET_GDRIVE_ID = "https://drive.google.com/file/d/1LQZ_J3ttMR56dcibRtLuoSHBC1GByAQR/view?usp=sharing" # <-- GANTI DENGAN ID ANDA
ALEXNET_MODEL_PATH = "AlexNet.h5"
LENET_MODEL_PATH = "LeNet-5.h5"

# Panggil fungsi download untuk model besar
download_model_from_gdrive(ALEXNET_GDRIVE_ID, ALEXNET_MODEL_PATH)

# --- 1. KONFIGURASI HALAMAN & GAYA ---
st.set_page_config(
    page_title="Analisis Visual CNN | Final Layout",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>
    .st-emotion-cache-1r6slb0 { /* Target spesifik untuk container dengan border */
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        padding: 1.5rem;
    }
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #28a745; /* Warna hijau untuk angka prediksi */
    }
    /* Sedikit ruang antar container di kolom kanan */
    .st-emotion-cache-1r6slb0:not(:first-child) {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI-FUNGSI HELPER DENGAN CACHING ---

@st.cache_resource
def load_models():
    """Memuat model dari file .h5 dengan caching."""
    try:
        # Gunakan variabel path yang sudah didefinisikan
        lenet_model = keras.models.load_model(LENET_MODEL_PATH)
        alexnet_model = keras.models.load_model(ALEXNET_MODEL_PATH)
        return lenet_model, alexnet_model
    except Exception as e:
        st.error(f"❌ **Error Memuat Model:** {e}", icon="🚨")
        return None, None

def create_prob_chart(probs, title, color_scale):
    """Membuat grafik batang probabilitas dengan Plotly."""
    df = pd.DataFrame({'Digit': [str(i) for i in range(10)], 'Confidence': probs * 100})
    fig = px.bar(
        df, x='Digit', y='Confidence', title=title,
        labels={'Confidence': 'Tingkat Keyakinan (%)'},
        text_auto='.2f', height=350, template='plotly_dark',
        color='Confidence', color_continuous_scale=color_scale
    )
    fig.update_layout(
        yaxis_range=[0, 115], title_font_size=18, xaxis_title=None,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False
    )
    fig.update_traces(textposition='outside')
    return fig

@st.cache_data
def get_visualized_activations(_model_name, layer_names, input_image_bytes):
    """Fungsi terpisah untuk kalkulasi dan plotting yang berat, lalu hasilnya di-cache."""
    lenet_m, alexnet_m = load_models()
    model = alexnet_m if _model_name == 'alexnet' else lenet_m
    
    if _model_name == 'alexnet':
        input_image = np.frombuffer(input_image_bytes, dtype=np.float32).reshape(1, 227, 227, 3)
    else:
        input_image = np.frombuffer(input_image_bytes, dtype=np.float32).reshape(1, 32, 32, 1)

    activation_model = tf.keras.Model(
        inputs=model.inputs, outputs=[model.get_layer(name).output for name in layer_names]
    )
    tf.get_logger().setLevel('ERROR')
    activations = activation_model.predict(input_image)
    tf.get_logger().setLevel('INFO')

    plt.style.use('dark_background')
    figs = []
    if not isinstance(activations, list): activations = [activations]
    
    for layer_name, layer_activation in zip(layer_names, activations):
        if not hasattr(layer_activation, 'shape') or len(layer_activation.shape) != 4: continue
        n_features = layer_activation.shape[-1]
        if n_features is None: continue
            
        n_cols = 8; n_rows = math.ceil(n_features / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.5))
        fig.suptitle(f"Peta Aktivasi - {layer_name.upper()}", fontsize=16)
        
        if n_rows == 1 and n_cols == 1: axes = np.array([axes])
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < n_features:
                ax.imshow(layer_activation[0, :, :, i], cmap='viridis')
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        figs.append(fig)
        
    return figs

# --- 3. STRUKTUR UTAMA APLIKASI ---

if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

lenet_model, alexnet_model = load_models()

st.title("🎯 Analisis Visual CNN")
st.markdown("Gambar sebuah angka, lalu tekan tombol analisis untuk membandingkan cara kerja model LeNet-5 dan AlexNet secara visual.")
st.divider()

col_input, col_results = st.columns([1, 1.2], gap="large")

with col_input:
    with st.container(border=True):
        st.subheader("🎨 Kanvas Gambar", anchor=False)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)", stroke_width=22, stroke_color="#FFFFFF",
            background_color="#222", height=350, width=350,
            drawing_mode="freedraw", key="canvas",
        )
        predict_button = st.button("🚀 Analisis Gambar Ini", use_container_width=True, type="primary")

    with st.container(border=True):
        st.subheader("📜 Riwayat Prediksi", anchor=False)
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada riwayat prediksi.")

# --- PERUBAHAN UTAMA ADA DI BAGIAN INI ---
with col_results:
    # Logika tombol HANYA untuk memicu prediksi dan menyimpannya di session_state
    if predict_button:
        if canvas_result.image_data is not None and lenet_model and alexnet_model:
            with st.spinner('Menganalisis gambar...'):
                img_gray = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
                
                # Proses & prediksi
                img_lenet_processed = np.reshape(cv2.resize(img_gray, (32, 32)), (1, 32, 32, 1)).astype('float32') / 255.0
                pred_lenet_prob = lenet_model.predict(img_lenet_processed)[0]
                
                img_alexnet_processed = np.reshape(cv2.cvtColor(cv2.resize(img_gray, (227, 227)), cv2.COLOR_GRAY2RGB), (1, 227, 227, 3)).astype('float32') / 255.0
                pred_alexnet_prob = alexnet_model.predict(img_alexnet_processed)[0]

                # Simpan semua hasil ke session_state
                st.session_state.last_prediction = {
                    "lenet_prob": pred_lenet_prob,
                    "alexnet_prob": pred_alexnet_prob,
                    "img_lenet_bytes": img_lenet_processed.tobytes(),
                    "img_alexnet_bytes": img_alexnet_processed.tobytes()
                }
                
                # Update riwayat
                new_entry = {
                    "Prediksi LeNet-5": np.argmax(pred_lenet_prob), "Keyakinan L5 (%)": f"{np.max(pred_lenet_prob)*100:.1f}",
                    "Prediksi AlexNet": np.argmax(pred_alexnet_prob), "Keyakinan AN (%)": f"{np.max(pred_alexnet_prob)*100:.1f}",
                }
                if not st.session_state.history or st.session_state.history[0] != new_entry:
                    st.session_state.history.insert(0, new_entry)
                
                st.rerun()
        else:
            st.warning("Gambar sesuatu di kanvas terlebih dahulu!")

    # Logika tampilan HANYA membaca dari session_state
    if st.session_state.last_prediction:
        # Unpack data dari session state
        lenet_prob = st.session_state.last_prediction["lenet_prob"]
        alexnet_prob = st.session_state.last_prediction["alexnet_prob"]
        lenet_digit = np.argmax(lenet_prob)
        alexnet_digit = np.argmax(alexnet_prob)
        
        # --- BAGIAN 1: RINGKASAN PREDIKSI (BARU) ---
        with st.container(border=True):
            st.subheader("💡 Ringkasan Prediksi", anchor=False)
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.metric(
                    label="Prediksi LeNet-5",
                    value=f"{lenet_digit}",
                    help=f"Tingkat keyakinan: {np.max(lenet_prob)*100:.1f}%"
                )
            with col_sum2:
                st.metric(
                    label="Prediksi AlexNet",
                    value=f"{alexnet_digit}",
                    help=f"Tingkat keyakinan: {np.max(alexnet_prob)*100:.1f}%"
                )

        # --- BAGIAN 2: ANALISIS MENDALAM (LAMA, SEKARANG DI BAWAH) ---
        with st.container(border=True):
            st.subheader("📊 Analisis Mendalam", anchor=False)
            tab1, tab2 = st.tabs(["Detail LeNet-5", "Detail AlexNet"])
            with tab1:
                # Grafik probabilitas
                st.plotly_chart(create_prob_chart(lenet_prob, "Distribusi Keyakinan LeNet-5", 'Blues'), use_container_width=True)
                # Peta Aktivasi
                with st.expander("Lihat Peta Aktivasi LeNet-5"):
                    layer_names = [l.name for l in lenet_model.layers if 'conv' in l.name]
                    figs = get_visualized_activations('lenet', layer_names, st.session_state.last_prediction["img_lenet_bytes"])
                    for fig in figs: st.pyplot(fig); plt.close(fig)

            with tab2:
                # Grafik probabilitas
                st.plotly_chart(create_prob_chart(alexnet_prob, "Distribusi Keyakinan AlexNet", 'Greens'), use_container_width=True)
                # Peta Aktivasi
                with st.expander("Lihat Peta Aktivasi AlexNet"):
                    layer_names = [l.name for l in alexnet_model.layers if 'conv' in l.name][:3]
                    figs = get_visualized_activations('alexnet', layer_names, st.session_state.last_prediction["img_alexnet_bytes"])
                    for fig in figs: st.pyplot(fig); plt.close(fig)
    else:
        # Placeholder jika belum ada prediksi
        with st.container(border=True):
             st.info("Hasil analisis akan muncul di sini setelah Anda menggambar dan menekan tombol.", icon="💡")