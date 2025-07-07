# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
import cv2
from collections import Counter
import gdown

st.set_page_config(page_title="APP")

@st.cache_resource

def download_model_from_drive():
    drive_file_id = "1wFxvPmNnD70acFwWEIyuLfRRH1ONKtc5"
    output_path = "segmen_LR_0.0001_25_vgg19_fold_2.keras"
    url = f"https://drive.google.com/uc?id={drive_file_id}"
    
    if not os.path.exists(output_path):
        with st.spinner("🔽 Mengunduh model dari Google Drive..."):
            gdown.download(url, output_path, quiet=False)
            st.success("✅ Model berhasil diunduh!")
    else:
        st.info("📦 Model sudah tersedia secara lokal.")

def load_keras_model():
    model_path = 'segmen_LR_0.0001_25_vgg19_fold_2.keras'
    
    # Unduh jika belum ada
    download_model_from_drive()

    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di path: {model_path}")
        return None

    model = load_model(model_path)
    return model

def segmented_hsv(hsv_img, index):
    if index == 0:
        lower = np.array([20,20,0], dtype=np.uint8)
        upper = np.array([89,255,255], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower, upper)
    elif index == 1:
        lower = np.array([28,25,25], dtype=np.uint8)
        upper = np.array([100,255,255], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower, upper)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    elif index == 2 or index == 3:
        lower = np.array([20,80,80], dtype=np.uint8) if index == 2 else np.array([20,40,40], dtype=np.uint8)
        upper = np.array([100,255,255], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower, upper)
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
    else:
        raise ValueError("Indeks threshold tidak valid (harus 0-3).")
    return mask

def majority_vote(predictions):
    count = Counter(predictions)
    most_common = count.most_common(1)[0]
    return most_common[0], most_common[1] / len(predictions)

def main():
    st.markdown("""
    <h1 style='text-align: left; font-size: 28px; '>
        Klasifikasi Penyakit Pada Citra Daun Jagung Menggunakan Arsitektur VGG-19 Dengan Segmentasi Citra HSV (Hue, Saturation, Value)
    </h1>
    <style>
    .sidebar-logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px 0;
    }
    .sidebar-logo-container img {
        width: 150px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.image("logo utm.png", width=100)
    selected_tab = st.sidebar.radio("Menu", ["Data Understanding", "Prediction"])

    if selected_tab == "Data Understanding":
        st.header("Data Understanding")
        st.write('Data yang digunakan pada klasifikasi ini adalah data citra daun jagung')
        st.markdown("""
        Link Dataset:
        https://www.kaggle.com/datasets/ndisan/corn-leaf-disease
        """, unsafe_allow_html=True)

        st.write("Sampel Data : ")
        DATA_DIR = "data"
        class_descriptions = {
            "bercak_daun": (
                "bercak hitam atau cokelat gelap : disebabkan oleh penyakit busuk daun atau juga terserang jamur."
                "bercak kuning atau kecokelatan : ditandai sebagai tanaman yang kekurangan nutrisi, seperti kekurangan zat besi atau magnesium "
            ),
            "daun_sehat": (
                "Tanaman jagung yang memiliki daun sehat dapat dilihat dengan ciri memiliki pelepah, tangkai, dan helai daun yang bentuknya memanjang dan bersih bewarna hijau dengan tulang daun yang sejajar ibu daun. "

            ),
            "hawar_daun": (
                "Tanda awalnya muncul gejala bercak kecil berwarna cokelat kehijauan berbentuk bulat memanjang, kemudian berkembang menjadi lebih besar."
                "Disebabkan oleh *miselium* jamur *E.turcium*"
                "Ketika infeksi sudah terlalu parah, daun akan mengalami nekrosis dan bercak akan mengering"
            ),
            "karat_daun": (
                "Diawali dengan munculnya bercak kecil (*uredinia*) yang menghasilkan *eredospore* berbentuk oval yang kemudian semakin memanjang."
                "Berwarna hijau keabu-abuan hingga coklat/kering."
                "Infeksi dapat menjalar ke tanaman jagung lain melalui perantara angin."
            )
        }
        classes = list(class_descriptions.keys())
        for cls in classes:
            st.write(f"Kelas: {cls.replace('_', ' ')}")
            class_path = os.path.join(DATA_DIR, cls)
            if os.path.exists(class_path):
                img_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))][:2]
                cols = st.columns(2)
                for i, img_file in enumerate(img_files):
                    img_path = os.path.join(class_path, img_file)
                    img = Image.open(img_path)
                    cols[i].image(img, caption=img_file, width=300)
                st.markdown(f"**Penjelasan:** {class_descriptions[cls]}")
            else:
                st.warning(f"Folder untuk kelas {cls} tidak ditemukan.")

    elif selected_tab == "Prediction":
        st.header("Prediction")
        st.write("Unggah gambar untuk diklasifikasikan")
        st.write("🔄 Memuat model...")
        model = load_keras_model()
        st.success(" Model berhasil dimuat!")

        upload_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

        if upload_file is not None:
            try:
                image = Image.open(upload_file).convert("RGB")

                img_resized = image.resize((224, 224))

                img_np = np.array(img_resized)
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)


                # Pisahkan channel HSV
                h, s, v = cv2.split(hsv)

                # Gabungkan kembali sebagai citra 3-channel pseudo-RGB, tetap dalam HSV
                # untuk ditampilkan sebagai citra warna palsu (false-color)
                hsv_stack = np.stack([h, s, v], axis=-1)  # shape: (height, width, 3)

                # Pastikan data dalam format uint8 (0-255) agar bisa ditampilkan sebagai gambar berwarna
                hsv_image = Image.fromarray(hsv_stack, mode='RGB')


                # Tampilkan hasil proses gambar
                st.subheader("1. Visualisasi Proses Gambar")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Gambar Input (RGB Asli)**")
                    st.image(image, caption="Input Asli", use_container_width=True)

                with col2:
                    st.markdown("**Setelah Resize (224x224)**")
                    st.image(img_resized, caption="Hasil Resize", use_container_width=True)

                with col3:
                    st.markdown("**HSV (Gabungan H, S, V)**")
                    st.image(hsv_image, caption="HSV Gabungan", use_container_width=True)

                st.subheader("2. Hasil Prediksi")
                segmented_images = []
                predicted_classes = []
                predicted_confidences = []
                cols = st.columns(4)
                labels = ['Bercak Daun', 'Daun Sehat', 'Hawar Daun', 'Karat Daun']
                

                for i in range(4):
                    mask = segmented_hsv(hsv, i)
                    segmented = cv2.bitwise_and(img_np, img_np, mask=mask)
                    segmented_images.append(segmented)

                    img_array = segmented / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    prediction = model.predict(img_array)
                    pred_class = np.argmax(prediction)
                    confidence = np.max(prediction)
                    predicted_classes.append(pred_class)
                    predicted_confidences.append(confidence)

                    seg_pil = Image.fromarray(segmented)
                    
                    cols[i].image(seg_pil, caption=f"Threshold {i+1} — {labels[pred_class]} ({confidence:.2f})", use_container_width=True)

                    # Evaluasi hasil akhir berdasarkan kombinasi voting dan confidence
                unique_preds = set(predicted_classes)

                if len(unique_preds) == 4:
                    # Semua hasil berbeda → ambil yang confidence tertinggi
                    max_conf_index = np.argmax(predicted_confidences)
                    final_class = predicted_classes[max_conf_index]
                    final_label = labels[final_class]
                    final_confidence = predicted_confidences[max_conf_index]
                    st.warning(f"Semua prediksi berbeda — dipilih kelas dengan akurasi tertinggi: {final_label} ({final_confidence:.2f})")
                else:
                    # Hitung jumlah vote dan rata-rata confidence per kelas
                    class_conf_dict = {i: [] for i in range(len(labels))}
                    for cls, conf in zip(predicted_classes, predicted_confidences):
                        class_conf_dict[cls].append(conf)

                    combined_scores = {}
                    for cls, conf_list in class_conf_dict.items():
                        if conf_list:
                            avg_conf = np.mean(conf_list)
                            vote_count = len(conf_list)
                            combined_scores[cls] = (vote_count, avg_conf)

                    # Urutkan berdasarkan jumlah vote → lalu confidence rata-rata
                    final_class = sorted(combined_scores.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)[0][0]
                    final_label = labels[final_class]
                    final_vote, final_avg_conf = combined_scores[final_class]
                    st.success(f"Hasil akhir kelas dominan dan confidence: {final_label} "
                                f"(Dipilih {final_vote}/4, rata-rata akurasi: {final_avg_conf:.2f})")


            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

if __name__ == "__main__":
    main()
