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
from datetime import datetime
import time

# --- 1. KONFIGURASI DAN GAYA ---
st.set_page_config(
    page_title="CNN Visual Analysis | Enhanced Edition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    """Membaca file CSS dan menyuntikkannya ke dalam aplikasi Streamlit."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"❌ **File CSS tidak ditemukan:** Pastikan file '{file_name}' ada di direktori yang sama dengan `app.py`.")

load_css("style.css")


# --- 2. SIDEBAR - INFORMASI DAN PANDUAN ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">🧠 CNN Explorer</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Deep Learning Made Visual</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ **Tentang Aplikasi**", expanded=True):
        st.markdown("""
        **CNN Visual Analysis** adalah aplikasi interaktif yang memungkinkan Anda memahami cara kerja Neural Network secara visual.

        **🎯 Tujuan:**
        - Membandingkan performa LeNet-5 vs AlexNet
        - Visualisasi proses pembelajaran AI
        - Edukasi Deep Learning untuk semua level

        **🔬 Model yang Digunakan:**
        - **LeNet-5**: Pionir CNN (1998)
        - **AlexNet**: Revolusi Deep Learning (2012)
        """)

    with st.expander("🚀 **Panduan Cepat**"):
        st.markdown("""
        **Untuk Pemula:**
        1. 🎨 Gambar angka 0-9 di kanvas
        2. 🚀 Klik "Analisis Gambar"
        3. 📊 Lihat prediksi kedua model
        4. 🔍 Eksplorasi peta aktivasi

        **Untuk Expert:**
        - Bandingkan arsitektur CNN
        - Analisis feature maps
        - Evaluasi confidence scores
        - Studi kasus preprocessing
        """)

    with st.expander("⚙️ **Spesifikasi Teknis**"):
        st.markdown("""
        **Model Specifications:**
        - **LeNet-5**: 32×32, Grayscale
        - **AlexNet**: 227×227, RGB

        **Preprocessing:**
        - Normalization: [0, 1]
        - Resize otomatis
        - Color space conversion

        **Visualization:**
        - Conv layer activation maps
        - Confidence distribution
        - Real-time processing
        """)

    with st.expander("👥 **Credits & Acknowledgments**"):
        st.markdown("""
        **🤖 AI Assistants:**
        - **Gemini**: Model architecture insights
        - **GPT**: Code optimization & UI design

        **📚 Frameworks:**
        - TensorFlow/Keras
        - Streamlit
        - OpenCV
        - Plotly

        **🎓 Educational Purpose:**
        Aplikasi ini dibuat untuk tujuan edukasi dan penelitian dalam bidang Computer Vision dan Deep Learning.
        """)

    st.markdown("---")
    user_level = st.selectbox(
        "👤 **Level Pengalaman Anda:**",
        ["🔰 Pemula (Baru kenal AI)", "🎓 Menengah (Punya dasar ML)", "🚀 Expert (Deep Learning Pro)"],
        help="Pilih level untuk mendapatkan penjelasan yang sesuai"
    )

# --- 3. FUNGSI-FUNGSI BANTU ---
ALEXNET_MODEL_PATH = "AlexNet.h5"
LENET_MODEL_PATH = "LeNet-5.h5"

@st.cache_resource
def load_models():
    """Memuat model dari file .h5 lokal dengan caching."""
    try:
        lenet_model = keras.models.load_model(LENET_MODEL_PATH)
        alexnet_model = keras.models.load_model(ALEXNET_MODEL_PATH)
        return lenet_model, alexnet_model
    except Exception as e:
        if "No such file or directory" in str(e):
            st.error(
                f"❌ **Error Model Tidak Ditemukan:** Pastikan file '{LENET_MODEL_PATH}' dan '{ALEXNET_MODEL_PATH}' "
                "ada di repository Anda dan telah di-push dengan benar (disarankan menggunakan Git LFS).",
                icon="🚨"
            )
        else:
            st.error(f"❌ **Error Memuat Model:** {e}", icon="🚨")
        return None, None

def create_prob_chart(probs, title, color_scale):
    """Membuat grafik batang probabilitas dengan Plotly."""
    df = pd.DataFrame({'Digit': [str(i) for i in range(10)], 'Confidence': probs * 100})
    fig = px.bar(
        df, x='Digit', y='Confidence', title=title,
        labels={'Confidence': 'Tingkat Keyakinan (%)'},
        text_auto='.1f', height=400, template='plotly_dark',
        color='Confidence', color_continuous_scale=color_scale
    )
    fig.update_layout(
        yaxis_range=[0, 105], title_font_size=20, xaxis_title="Digit",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.1)',
        coloraxis_showscale=False, font_size=14
    )
    fig.update_traces(textposition='outside', textfont_size=12)
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 2))
        fig.suptitle(f"🔍 Feature Maps - {layer_name.upper()}", fontsize=18, color='white')

        if n_rows == 1 and n_cols == 1: axes = np.array([axes])
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < n_features:
                ax.imshow(layer_activation[0, :, :, i], cmap='viridis')
                ax.set_title(f'Filter {i+1}', fontsize=10, color='white')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        figs.append(fig)

    return figs

def get_explanation_by_level(level):
    """Memberikan penjelasan sesuai level pengguna."""
    if "Pemula" in level:
        return {
            "prediction": "🎯 **Prediksi** adalah tebakan terbaik model tentang angka yang Anda gambar.",
            "confidence": "📊 **Keyakinan** menunjukkan seberapa yakin model dengan tebakannya (0-100%).",
            "activation": "🗺️ **Peta Aktivasi** seperti 'mata' model yang melihat bagian-bagian penting dari gambar Anda."
        }
    elif "Menengah" in level:
        return {
            "prediction": "🎯 **Prediction** menggunakan softmax untuk klasifikasi multi-class dengan 10 output nodes.",
            "confidence": "📊 **Confidence Score** dari softmax probability distribution, menunjukkan probabilitas untuk setiap kelas.",
            "activation": "🗺️ **Activation Maps** menvisualisasikan feature detection pada setiap convolutional layer."
        }
    else:  # Expert
        return {
            "prediction": "🎯 **Classification Output** dengan argmax dari softmax layer, optimized menggunakan categorical crossentropy.",
            "confidence": "📊 **Posterior Probability** dari softmax function: σ(zi) = e^zi / Σe^zj, interpretable sebagai class confidence.",
            "activation": "🗺️ **Feature Maps** dari conv layers menunjukkan learned representations dan spatial feature detection patterns."
        }

# --- 4. MAIN APPLICATION ---

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Load models
lenet_model, alexnet_model = load_models()

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 CNN Visual Analysis</h1>
    <p>Explore Deep Learning Through Interactive Visualization</p>
</div>
""", unsafe_allow_html=True)

# Get explanations based on user level
explanations = get_explanation_by_level(user_level)

# Main content area
col_input, col_results = st.columns([1, 1.3], gap="large")

with col_input:
    # Drawing Canvas
    with st.container():
        st.markdown("### 🎨 **Drawing Canvas**")
        st.markdown(explanations["prediction"])

        st.markdown('<div>', unsafe_allow_html=True)
        
        # Gunakan session state untuk mengontrol key canvas
        if 'canvas_key' not in st.session_state:
            st.session_state.canvas_key = "canvas"
        
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=25,
            stroke_color="#FFFFFF",
            background_color="#1a1a1a",
            height=350,
            width=350,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            predict_button = st.button("🚀 **Analisis Gambar Ini**", use_container_width=True, type="primary")
        with col_btn2:
            clear_button = st.button("🗑️", help="Clear Canvas & Results", use_container_width=True)

        # === LOGIC TOMBOL CLEAR YANG DIPERBAIKI ===
        if clear_button:
            if 'last_prediction' in st.session_state:
                del st.session_state['last_prediction']
            # Reset canvas dengan mengubah key-nya
            st.session_state.canvas_key = f"canvas_{int(time.time())}"
            st.rerun()

    # History Section
    with st.container():
        st.markdown("### 📜 **Riwayat Prediksi**")
        if st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history, use_container_width=True, hide_index=True)

            if len(st.session_state.history) > 1:
                st.markdown("**📈 Statistik Singkat:**")
                col_stat1, col_stat2 = st.columns(2)
                try:
                    with col_stat1:
                        avg_lenet = np.mean([float(str(h["Keyakinan L5 (%)"]).replace('%', '')) for h in st.session_state.history])
                        st.metric("Rata-rata LeNet-5", f"{avg_lenet:.1f}%")
                    with col_stat2:
                        avg_alexnet = np.mean([float(str(h["Keyakinan AN (%)"]).replace('%', '')) for h in st.session_state.history])
                        st.metric("Rata-rata AlexNet", f"{avg_alexnet:.1f}%")
                except (ValueError, KeyError) as e:
                     st.warning("Gagal menghitung statistik rata-rata.", icon="⚠️")

        else:
            st.info("💡 Mulai menggambar untuk melihat riwayat prediksi!", icon="📝")

with col_results:
    if predict_button:
        if canvas_result.image_data is not None and lenet_model and alexnet_model:
            progress_bar = st.progress(0, text="Menginisialisasi analisis...")
            
            with st.spinner('🔄 Memproses gambar dan menjalankan model...'):
                # Preprocessing
                progress_bar.progress(20, text="Preprocessing gambar...")
                img_gray = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)

                # Process for LeNet-5
                progress_bar.progress(40, text="Menjalankan LeNet-5...")
                img_lenet_processed = np.reshape(cv2.resize(img_gray, (32, 32)), (1, 32, 32, 1)).astype('float32') / 255.0
                pred_lenet_prob = lenet_model.predict(img_lenet_processed)[0]

                # Process for AlexNet
                progress_bar.progress(70, text="Menjalankan AlexNet...")
                img_alexnet_processed = np.reshape(cv2.cvtColor(cv2.resize(img_gray, (227, 227)), cv2.COLOR_GRAY2RGB), (1, 227, 227, 3)).astype('float32') / 255.0
                pred_alexnet_prob = alexnet_model.predict(img_alexnet_processed)[0]
                
                progress_bar.progress(90, text="Menyimpan hasil...")
                st.session_state.last_prediction = {
                    "lenet_prob": pred_lenet_prob,
                    "alexnet_prob": pred_alexnet_prob,
                    "img_lenet_bytes": img_lenet_processed.tobytes(),
                    "img_alexnet_bytes": img_alexnet_processed.tobytes(),
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }

                new_entry = {
                    "Waktu": st.session_state.last_prediction["timestamp"],
                    "Prediksi LeNet-5": np.argmax(pred_lenet_prob),
                    "Keyakinan L5 (%)": f"{np.max(pred_lenet_prob)*100:.1f}%",
                    "Prediksi AlexNet": np.argmax(pred_alexnet_prob),
                    "Keyakinan AN (%)": f"{np.max(pred_alexnet_prob)*100:.1f}%",
                }
                
                # Menghindari duplikat jika tombol ditekan berkali-kali tanpa menggambar ulang
                if not st.session_state.history or st.session_state.history[0]["Waktu"] != new_entry["Waktu"]:
                    st.session_state.history.insert(0, new_entry)
                    if len(st.session_state.history) > 10:
                        st.session_state.history = st.session_state.history[:10]
                
                progress_bar.progress(100, text="Analisis selesai!")
                st.success("✅ Analisis selesai!", icon="🎉")
                st.rerun()
        else:
            st.warning("⚠️ Silakan gambar sesuatu di kanvas terlebih dahulu!", icon="✏️")
    
    # Display Results
    if st.session_state.last_prediction:
        lenet_prob = st.session_state.last_prediction["lenet_prob"]
        alexnet_prob = st.session_state.last_prediction["alexnet_prob"]
        lenet_digit = np.argmax(lenet_prob)
        alexnet_digit = np.argmax(alexnet_prob)
        lenet_conf = np.max(lenet_prob)
        alexnet_conf = np.max(alexnet_prob)

        # Results Summary
        with st.container():
            st.markdown("### 🎯 **Hasil Prediksi**")
            st.markdown(explanations["confidence"])
            
            col_sum1, col_sum2, col_sum3 = st.columns([1, 1, 1])
            
            with col_sum1:
                st.metric(
                    label="🥇 LeNet-5 (Pioneer)", value=f"{lenet_digit}",
                    delta=f"{lenet_conf*100:.1f}% confidence",
                    help="Model CNN pertama yang sukses (1998)"
                )
            
            with col_sum2:
                st.metric(
                    label="🚀 AlexNet (Modern)", value=f"{alexnet_digit}",
                    delta=f"{alexnet_conf*100:.1f}% confidence",
                    help="Revolusi Deep Learning (2012)"
                )
            
            with col_sum3:
                if lenet_conf > alexnet_conf:
                    winner = "LeNet-5"
                    diff = (lenet_conf - alexnet_conf) * 100
                elif alexnet_conf > lenet_conf:
                    winner = "AlexNet"
                    diff = (alexnet_conf - lenet_conf) * 100
                else:
                    winner = "Seri"
                    diff = 0
                
        # Detailed Analysis
        with st.container():
            st.markdown("### 📊 **Analisis Mendalam**")
            st.markdown(explanations["activation"])
            
            tab1, tab2, tab3 = st.tabs(["🔍 LeNet-5 Analysis", "🚀 AlexNet Analysis", "⚖️ Model Comparison"])
            
            with tab1:
                st.plotly_chart(create_prob_chart(lenet_prob, "LeNet-5 Confidence Distribution", 'Blues'), use_container_width=True)
                with st.expander("🗺️ **Feature Maps LeNet-5**"):
                    st.markdown("LeNet-5 menggunakan 2 layer konvolusi. Berikut adalah visualisasi bagaimana setiap filter mendeteksi fitur:")
                    layer_names = [l.name for l in lenet_model.layers if 'conv' in l.name]
                    if layer_names:
                        figs = get_visualized_activations('lenet', layer_names, st.session_state.last_prediction["img_lenet_bytes"])
                        for fig in figs:
                            st.pyplot(fig)
                            plt.close(fig)

            with tab2:
                st.plotly_chart(create_prob_chart(alexnet_prob, "AlexNet Confidence Distribution", 'Viridis'), use_container_width=True)
                with st.expander("🗺️ **Feature Maps AlexNet**"):
                    st.markdown("AlexNet menggunakan 5 layer konvolusi. Visualisasi ini menunjukkan 3 layer pertama untuk performa optimal:")
                    layer_names = [l.name for l in alexnet_model.layers if 'conv' in l.name][:3]
                    if layer_names:
                        figs = get_visualized_activations('alexnet', layer_names, st.session_state.last_prediction["img_alexnet_bytes"])
                        for fig in figs:
                            st.pyplot(fig)
                            plt.close(fig)
            
            with tab3:
                st.markdown("#### 🔬 **Perbandingan Mendalam**")
                comparison_data = {'Model': ['LeNet-5', 'AlexNet'], 'Confidence': [lenet_conf * 100, alexnet_conf * 100], 'Prediction': [lenet_digit, alexnet_digit]}
                fig_comparison = px.bar(
                    comparison_data, x='Model', y='Confidence', title='Perbandingan Keyakinan Model',
                    text='Prediction', height=400, color='Confidence', color_continuous_scale='RdYlBu_r', template='plotly_dark'
                )
                fig_comparison.update_traces(textposition='outside')
                fig_comparison.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.1)', showlegend=False)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                col_tech1, col_tech2 = st.columns(2)
                with col_tech1:
                    st.markdown("""**LeNet-5 Characteristics:**\n- 📅 Year: 1998\n- 🏗️ Layers: 5\n- 📏 Input: 32×32 Grayscale\n- ⚡ Parameters: ~60K\n- 🎯 Focus: Simple patterns""")
                with col_tech2:
                    st.markdown("""**AlexNet Characteristics:**\n- 📅 Year: 2012\n- 🏗️ Layers: 8\n- 📏 Input: 227×227 RGB\n- ⚡ Parameters: ~60M\n- 🎯 Focus: Complex features""")
                    
    else:
        # Placeholder
        with st.container():
            st.markdown("### 🎨 **Siap untuk Memulai?**")
            st.info(
                "✨ **Petunjuk Penggunaan:**\n\n"
                "1. 🖊️ Gambar angka (0-9) di kanvas sebelah kiri\n"
                "2. 🚀 Klik tombol 'Analisis Gambar Ini'\n"
                "3. 📊 Lihat hasil prediksi dari kedua model\n"
                "4. 🔍 Eksplorasi feature maps untuk memahami cara kerja AI\n\n"
                "💡 **Tips:** Gambar angka yang jelas dan berukuran cukup besar untuk hasil terbaik!",
                icon="🎯"
            )

# --- 6. FOOTER ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>🧠 CNN Visual Analysis</strong> | Enhanced Edition</p>
    <p>Built with ❤️ using Streamlit • TensorFlow • OpenCV • Plotly</p>
    <p><strong>AI Assistants:</strong> Gemini & GPT • <strong>Educational Purpose</strong> • <strong>Open Source</strong></p>
    <p>© 2025 | Deep Learning Made Visual | Version 2.1</p>
</div>
""", unsafe_allow_html=True)