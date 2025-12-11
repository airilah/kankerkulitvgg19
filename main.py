import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from PIL import Image
import altair as alt
import onnxruntime as ort

# Konfigurasi Streamlit
st.set_page_config(page_title="Klasifikasi Kanker Kulit", layout="wide")

# --- Sidebar Menu ---
menu = st.sidebar.radio("Pilih Menu", [
    "Ringkasan Penelitian",
    "Program Penelitian",
    "Prediksi Gambar"
])

# --- 1. Ringkasan Penelitian ---
if menu == "Ringkasan Penelitian":
    st.title("📄 RINGKASAN PENELITIAN")
    st.header("PENERAPAN RANDOM OVER SAMPLING PADA PENYAKIT KANKER KULIT MELALUI KLASIFIKASI CITRA MENGGUNAKAN ARSITEKTUR VGG19")
    st.header("BAB I PENDAHULUAN")
    st.subheader("Latar Belakang")
    st.markdown('''
    Kanker kulit adalah salah satu penyakit dengan pertumbuhan tercepat di dunia, dengan lebih dari satu juta kasus
    yang dilaporkan secara global pada tahun 2018. Paparan sinar ultraviolet, merokok, perubahan lingkungan, virus,
    dan konsumsi alkohol merupakan penyebab penyakit kanker kulit. Menurut data dari WHO, kasus melanoma yang
    terdiagnosis setiap tahun meningkat sebesar 53%, dengan tingkat kematian yang juga terus meningkat. Salah satu
    penyebab rendahnya tingkat kesembuhan adalah kesalahan dalam diagnosis dini, yang menyebabkan tingkat kelangsungan
    hidup kurang dari 14%. Namun, jika kanker kulit dapat dideteksi sejak dini, tingkat kelangsungan hidup dapat
    meningkat hingga 97%.

    Dalam klasifikasi citra menggunakan dataset HAM10000, terdapat tantangan berupa ketidakseimbangan jumlah data di
    setiap kelas. Salah satu cara untuk mengatasi masalah ini adalah dengan menerapkan teknik oversampling, yang
    bertujuan untuk meningkatkan jumlah sampel pada kelas minoritas dengan menduplikasi data. Penelitian ini juga
    memanfaatkan metode transfer learning sebagai peningkatan performa kinerja model. Diharapkan penelitian ini mampu
    menghasilkan akurasi yang tinggi dengan performa model yang lebih baik untuk menurunkan tingkat kematian dan
    meningkatkan tingkat kesembuhan pasien.
    ''')
    st.subheader("Permasalahan")
    st.markdown('''
    Dataset HAM10000 Skin Cancer ini mengalami ketidakseimbangan antara kelasnya yang dimana terdapat kelas mayoritas
    yang memiliki jauh lebih banyak daripada kelas minoritas. Kemudian hasil akurasi yang kurang baik dikarenakan kurangnya
    ada pengoptimalan terhadap model arsitektur yang dipakai.
    ''')
    st.subheader("Solusi Masalah")
    st.markdown('''
    Solusi yang diusulkan dengan penerapan random over sampling (ROS) yang dimana melakukan penambahan data baru dari kelas
    minoritas agar menyeimbangi dengan kelas mayoritas. Teknik ini dapat memiliki kemampuan untuk menghasilkan akurasi yang
    lebih optimal sehingga kinerja model lebih meningkat. Tak hanya itu, dengan melakukan penambahan menggunakan Transfer learning
    model pretrained dari ImageNet akan menjadi solusi untuk dapat menghasilkan kinerja dan akurasi yang lebih baik.
    ''')
    st.subheader("Pertanyaan")
    st.markdown('''
    Bagaimana pengaruh penerapan teknik Random Over Sampling (ROS) dan Transfer Learning berbasis ImageNet (dengan fine-tuning)
    terhadap kinerja dan waktu pelatihan model klasifikasi VGG19 pada dataset HAM10000?
    ''')
    st.subheader("Tujuan")
    st.markdown('''
    1. Mengetahui pengaruh penerapan teknik Random Over Sampling (ROS) dan Transfer Learning berbasis ImageNet dengan pendekatan fine-tuningterhadap kinerja model klasifikasi VGG19 pada dataset HAM10000.
    2. Menganalisis pengaruh teknik tersebut terhadap efisiensi waktu pelatihan model klasifikasi VGG19.
    ''')
    st.subheader("Manfaat")
    st.markdown('''
    1. Menambah pengetahuan mengenai penerapan teknik Random Over Sampling untuk penyeimbang dataset dan penerapan Transfer Learning pada klasifikasi citra menggunakan arsitektur VGG19.
    2. Dari hasil penelitian ini, peneliti lain dapat menjadikan penelitian ini sebagai pandangan atau referensi untuk melanjutkan dan mengembangkan penelitian lebih lanjut.
    ''')
    st.subheader("Batasan Masalah")
    st.markdown('''
    1. Dataset yang digunakan Skin Cancer MNIST: HAM10000 yang merupakan data kanker kulit citra dermatoskopik diperoleh dari website kaggle dengan link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000.
    2. Pada dataset terdapat 10015 citra dengan 7 kelas yaitu Actinic Keratoses, Basal Cell Carcinoma, Benign Keratosis-Like Lesions, Dermatofibroma, Melanocytic Nevi, Melanoma, dan Vascular Lesions.
    3. Mengubah ukuran citra yang awalnya 600 x 450 piksel menjadi 64 x 64 piksel, dikarenakan ukurannya terlalu besar dan memerlukan waktu proses yang sangat lama.
    ''')


    st.header("BAB II TINJAUAN PUSTAKA")

    st.header("BAB III METODE USULAN")
    st.subheader("Dataset")
    st.markdown('''
    Dataset Skin Cancer MNIST: HAM10000 ini diunduh dari website kaggle dengan ukurannya sebesar 6 GB. Dataset ini berisi dari kumpulan gambar dermatoskopik yang diperoleh dan disimpan dari berbagai sumber dan populasi,
    serta dikumpulkan untuk keperluan penelitian dalam bidang machine learning khususnya dalam mendeteksi kanker kulit secara otomatis HAM10000 menyatukan gambar dari beberapa database dermatologi internasional,
    terutama dari dua institusi besar Universitas Medis Vienna (Austria) dan Cliff Rosendahl (Australia). Gambar-gambar ini mencerminkan kondisi nyata yang biasa ditemui dalam praktik dermatologi, mencakup variasi anatomi,
    pencahayaan, dan jenis lesi kulit. Hasil akhir dari kumpulan data ini berjumlah 10015 gambar dermatoskopik yang berguna untuk set pelatihan dengan tujuan pembelajaran mesin. Komponen yang dataset ini berisi folder HAM10000_images_part_1
    dengan menampung 5000 gambar dermatoskopik dan HAM10000_images_part_2 dengan menampung 5015 gambar dermatoskopik. Ukuran gambar dari dataset ini yaitu 600x450 piksel dengan format gambar JPG.
    ''')
    image_path = 'dataset.png'
    st.image(Image.open(image_path), caption='Jumlah Data Dataset', width=800)
    st.markdown('''
    Untuk kelas diagnosis dari dataset ini terdiri dari 7 kelas yaitu:
    1. Nevus Melanositik (nv): Tahi lalat jinak.
    2. Melanoma (mel): Kanker kulit yang serius.
    3. Lesi Keratosis Jinak (bkl): Lesi kulit yang menyerupai kanker.
    4. Karsinoma Sel Basal (bcc): Kanker kulit dengan pertumbuhan lambat.
    5. Keratosis Aktinik (akiec): Prakanker kulit akibat paparan sinar UV.
    6. Lesi Vaskular (vasc): Kondisi yang melibatkan pembuluh darah.
    7. Dermatofibroma (df): Tumor jinak kecil di kulit.
    ''')
    image_path = 'citra.png'
    st.image(Image.open(image_path), caption='dataset 7 kelas', width=800)

    st.subheader("Rancangan Sistem")
    image_path = 'rancangan sistem.png'
    st.image(Image.open(image_path), caption='Rancangan Sistem', width=600)

    st.subheader("Arsitektur VGG19")
    image_path = 'vgg19.png'
    st.image(Image.open(image_path), caption='Arsitektur VGG19', width=800)
    st.markdown('''
    Visual Geometry Group 19 atau VGG19 adalah salah satu arsitektur jaringan saraf konvolusi (CNN) yang diperkenalkan oleh Simonyan dan Zisserman dalam makalah berjudul
    "Very Deep Convolutional Networks for Large-Scale Image Recognition" pada tahun 2014 [9]. Model ini dikembangkan sebagai bagian dari penelitian untuk kompetisi ImageNet
    Large Scale Visual Recognition Challenge (ILSVRC). Arsitektur ini dirancang untuk meningkatkan akurasi klasifikasi gambar dengan memperdalam jumlah lapisan jaringan.
    VGG19 memiliki total 19 lapisan yang terdiri dari 16 lapisan konvolusi dengan filter ukuran 3x3, 3 fully connected layer pada bagian akhir.
    ''')

    st.subheader("VGG19 Fine-tuning")
    image_path = 'fine-tuning.png'
    st.image(Image.open(image_path), caption='VGG19 Fine-tuning', width=800)
    st.markdown('''
    penerapan fine-tuning dari arsitektur VGG19 yang dimana pada 3 layer awal dilakukan pembekuan sehingga menyisahkan sejumlah trainable layer untuk pelatihan lagi.
    Pembekuan dilakukan mulai dari lapisan konvolusi 1 hingga lapisan max pooling 1. Pada proses freeze terjadi proses perhitungan dengan menggunakan bobot dari ImageNe
    tetapi tidak perlu training ulang yang artinya bobot tidak berubah hingga max pooling 1. Untuk proses unfreeze dilakukan proses perhitungan dengan menggunakan bobot yang
    di update dengan output yang sebelumnya.
    ''')

    st.subheader("Tabel Uji coba sistem")
    st.dataframe({
        'Skenario': ['Skenario 1', 'Skenario 2', 'Skenario 3', 'Skenario 4'],
        'Random Over Sampling': ['Tanpa', 'Pakai', 'Tanpa', 'Pakai'],
        'Transfer Learning ImageNet Fine-tuning': ['Tanpa', 'Tanpa', 'Pakai', 'Pakai'],
        'Learning Rate': ['0,001']*4,
        'Epoch': [30]*4,
        'Batch size': [64]*4,
        'Optimizer': ['Adam']*4,
        'Loss Function': ['Cross-Entropy Loss']*4,
        'Waktu Pelatihan': ['Detik']*4
    })

    st.header("BAB IV HASIL DAN PEMBAHASAN")
    st.header("Analisis Hasil Uji Coba")
    st.subheader("Tabel Pelatihan Skenario Uji Coba")
    st.dataframe({
        'Skenario': ['Skenario 1', 'Skenario 3', 'Skenario 4'],
        'Random Over Sampling': ['Tanpa', 'Tanpa', 'Pakai'],
        'Transfer Learning ImageNet Fine-tuning': ['Tanpa', 'Tanpa', 'Pakai', 'Pakai'],
        'Fold Terbaik': [5]*4,
        'Akurasi Pelatihan (%)': [70.18, 94.85, 99.78],
        'Loss Pelatihan (%)': [85.23, 19.44, 1.06],
        'Waktu Pelatihan (detik)': [1957.90, 1232.24, 6663.85],
        'Learning Rate': ['0,001']*4,
        'Epoch': [30]*4,
        'Batch Size': [64]*4,
        'Optimizer': ['Adam']*4,
        'Loss Function': ['Cross-Entropy Loss']*4
    })
    st.subheader("Tabel Pengujian Skenario Uji Coba")
    st.dataframe({
        'Skenario': ['Skenario 1', 'Skenario 3', 'Skenario 4'],
        'Random Over Sampling': ['Tanpa', 'Tanpa', 'Pakai'],
        'Transfer Learning ImageNet Fine-tuning': ['Tanpa', 'Pakai', 'Pakai'],
        'Akurasi Pengujian (%)': [70.04, 71.54, 97.05],
        'Precision (%)': [85.49, 76.20, 97.21],
        'Recall (%)': [59.71, 67.95, 96.88],
        'AUC (%)': [94.26, 94.00, 99.42],
        'Loss Pengujian (%)': [82.28, 111.04, 16.03]
    })

    st.subheader("Kesimpulan")
    st.markdown('''
    Berdasarkan hasil eksperimen, skenario terbaik diperoleh pada skenario 3, yaitu model yang menggunakan Random Over Sampling (ROS) dan Transfer Learning. Model ini mencapai akurasi pelatihan
    tertinggi sebesar 99,78% dengan loss terendah 1,06%, meskipun waktu pelatihannya cukup lama, yaitu 6663,85 detik. Pada saat pengujian, model tetap menunjukkan performa tinggi dengan
    akurasi 97,05%, precision 97,21%, recall 96,88%, AUC 99,42%, dan loss 16,03%. Skenario 2 yang menggunakan Transfer Learning tanpa ROS memperoleh akurasi pelatihan tinggi sebesar 94,85%,
    tetapi performa pengujian turun signifikan menjadi akurasi 71,54% dan loss 111,04% yang mengindikasikan kemungkinan overfitting. Sementara itu, skenario 1 yang tidak menggunakan ROS maupun
    Transfer Learning menghasilkan performa terendah. Selain itu, Transfer Learning terbukti mempercepat pelatihan, seperti pada skenario 2 dengan waktu pelatihan 1232,24 detik, lebih cepat dibandingkan
    skenario 1 1957,90 detik. Secara keseluruhan, Random Over Sampling membantu model mengenali kelas minoritas, dan Transfer Learning memperkuat ekstraksi fitur serta mempercepat pelatihan. Kombinasi keduanya,
    seperti pada skenario 3, terbukti menjadi pendekatan paling efektif untuk klasifikasi citra medis berbasis VGG19, terutama dalam menangani ketidakseimbangan data dan meningkatkan generalisasi model.
    ''')

    st.subheader("Saran")
    st.markdown('''
    1. Meskipun ROS efektif, teknik ini berisiko menyebabkan overfitting karena duplikasi data minoritas. Disarankan untuk mengeksplorasi metode yang lebih canggih seperti SMOTE atau ADASYN yang menghasilkan data sintetis.
    2. Augmentasi citra seperti rotasi, zoom, dan flip juga bisa diterapkan untuk meningkatkan keragaman data pelatihan dan mengurangi overfitting, sehingga model lebih adaptif terhadap variasi citra klinis.
    3. Mengingat VGG19 cukup berat secara komputasi, penelitian selanjutnya dapat mempertimbangkan arsitektur yang lebih efisien seperti EfficientNet, DenseNet, atau MobileNet.
    4. Untuk menguji kemampuan generalisasi, model perlu diuji pada dataset eksternal yang tidak digunakan dalam pelatihan.
    5. Selain akurasi, aspek interpretabilitas model penting untuk diteliti. Teknik seperti Grad-CAM atau LIME dapat membantu memahami area citra yang memengaruhi keputusan klasifikasi.
    ''')




# --- 2. Program Penelitian ---
elif menu == "Program Penelitian":
    st.title("💻 Program Penelitian")
    st.header("Import Library")
    st.code('''
    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools
    import time
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,Embedding
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU, LSTM,Bidirectional,Attention,Concatenate
    from tensorflow.keras import regularizers, optimizers,losses
    from tensorflow.keras.metrics import Accuracy,Recall,Precision,AUC,TruePositives,TrueNegatives,FalseNegatives,FalsePositives, SpecificityAtSensitivity,SensitivityAtSpecificity
    from tensorflow.keras import Model, Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.utils import img_to_array
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.applications import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input
    import sklearn
    import sklearn.metrics as m
    from sklearn import tree
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import LabelEncoder, label_binarize
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve, auc as sklearn_auc
    import imblearn
    from imblearn.over_sampling import RandomOverSampler
    import skimage.io
    import skimage.color
    import skimage.filters
    from collections import Counter
    from tqdm import tqdm
    ''', language='python')

    st.header("Load Data")
    st.code('''
    # Memanggil Semua Folder Gambar beserta dengan File Metadata nya juga
    image_folders = [
        '/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1',
        '/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2'
    ]
    metadata_path = '/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv'

    # Membaca file metadata
    metadata = pd.read_csv(metadata_path)

    # Mapping nilai kolom 'dx' sesuai dengan dictionary dx
    dx_mapping = {
        'nv': 'melanocytic nevi',
        'mel': 'melanoma',
        'bkl': 'benign keratosis-like lesions',
        'bcc': 'basal cell carcinoma',
        'vasc': 'vascular lesions',
        'akiec': 'actinic keratoses and intraepithelial carcinomae',
        'df': 'dermatofibroma'
    }

    # Mengubah nilai kolom 'dx' menggunakan mapping
    metadata['dx'] = metadata['dx'].map(dx_mapping)

    # Menampilkan 5 baris pertama untuk memastikan perubahan
    metadata.head()
    ''', language='python')

    st.header("Output:")
    st.dataframe({
        'lesion_id': ['HAM_0000118', 'HAM_0000118', 'HAM_0002730', 'HAM_0002730', 'HAM_0001466'],
        'image_id': ['ISIC_0027419', 'ISIC_0025030', 'ISIC_0026769', 'ISIC_0025661', 'ISIC_0031633'],
        'dx': ['benign keratosis-like lesions']*5,
        'dx_type': ['histo']*5,
        'age': [80.0, 80.0, 80.0, 80.0, 75.0],
        'sex': ['male']*5,
        'localization': ['scalp', 'scalp', 'scalp', 'scalp', 'ear']
    })

    st.header("Preprocessing")
    st.code('''
    # Mengumpulkan gambar dalam masing-masing folder dan menyesuaikan dengan label nya
    label_mapping = dict(zip(metadata['image_id'], metadata['dx']))
    image_data = []
    labels = []
    first_image_array = None
    first_image_found = False

    # Fungsi untuk resize gambar
    def resize_image(img_path, target_size=(64, 64)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        return img_array
    # Fungsi untuk transformasi gambar (split channel dan flatten)
    def transform_image(img_array):
        R = img_array[:, :, 0].flatten()
        G = img_array[:, :, 1].flatten()
        B = img_array[:, :, 2].flatten()
        reordered_array = np.concatenate([R, G, B])
        return reordered_array

    # Loop untuk memproses gambar
    for folder in image_folders:
        image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]

        for image_file in tqdm(image_files, desc=f"Processing images in {folder}"):
            img_path = os.path.join(folder, image_file)
            # --- Resize gambar ---
            img_array = resize_image(img_path)
            # --- Simpan gambar pertama ---

            if not first_image_found:
                first_image_array = img_array.copy()
                first_image_found = True
            # --- Ambil ID gambar dan Label ---
            image_id = os.path.splitext(image_file)[0]
            label = label_mapping.get(image_id)

            if label is not None:
                # --- Transformasi gambar ---
                transformed_array = transform_image(img_array)

                # --- Simpan hasil ---
                image_data.append(transformed_array)
                labels.append(label)

    # Setelah semua gambar diproses, tampilkan gambar pertama
    if first_image_array is not None:
        print("Nilai numerik dari saluran Red (R), Green (G), dan Blue (B) untuk piksel pertama:")
        print("R:", first_image_array[0, 0, 0])
        print("G:", first_image_array[0, 0, 1])
        print("B:", first_image_array[0, 0, 2])

        print("Bentuk Gambar didalam first_image_array:", first_image_array.shape)
        print("Nilai 5 pixel pertama dari masing-masing channel :")
        print("Red channel:", first_image_array[:5, :5, 0])
        print("Green channel:", first_image_array[:5, :5, 1])
        print("Blue channel:", first_image_array[:5, :5, 2])

        # Mengubah menjadi DataFrame
        image_data = np.array(image_data).astype(int)  # Ubah ke int
        labels = np.array(labels)

        data = pd.DataFrame(image_data)
        data['label'] = labels

        # Membaca kembali CSV untuk verifikasi
        # data.to_csv('/kaggle/working/data_gambar.csv', index=False)
        data.head()
    ''', language='python')

    st.header("Output:")
    st.dataframe({
        'pixel_0': [217, 209, 221, 151, 170],
        'pixel_1': [216, 208, 223, 149, 171],
        'pixel_2': [217, 203, 228, 148, 171],
        'pixel_3': [213, 212, 228, 145, 173],
        # ...
        # Tambahkan beberapa pixel kolom lagi jika ingin tampil lebih panjang
        'pixel_12287': [85, 150, 139, 103, 136],
        'label': [
            'melanocytic nevi',
            'melanocytic nevi',
            'melanocytic nevi',
            'benign keratosis-like lesions',
            'benign keratosis-like lesions'
        ]
    })

    st.code('''
    # Deklarasikan valiarbel untuk menyimpan data gambar dan label nya secara terpisah
    x = data.drop(columns = ['label'])
    y = data['label']
    print('fitur', x.shape)
    print('label',y.shape)

    ros = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = ros.fit_resample(x, y)

    y_final = to_categorical(y_resampled)
    x_resampled = np.array(x_resampled)
    y_final = np.array(y_final)

    # Split data menjadi training dan testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        x_resampled, y_final, test_size=0.2, random_state=1, stratify=y_final
    )


    # Membagi data training menjadi data final dan validasi
    X_train_final, X_val, Y_train_final, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=1, stratify=Y_train
    )

    # Cetak bentuk data setelah pembagian
    print("Training Data:", X_train_final.shape, Y_train_final.shape)
    print("Validation Data:", X_val.shape, Y_val.shape)
    print("Testing Data:", X_test.shape, Y_test.shape)
    ''', language='python')

    st.header("Output:")
    st.markdown('''
    Training Data: (30038, 12288) (30038, 7)
    Validation Data: (7510, 12288) (7510, 7)
    Testing Data: (9387, 12288) (9387, 7)
    ''')

    st.header("Mengubah Citra ke 3D")
    st.code('''
    # Pastikan data gambar direshape sesuai dimensi 64x64 dan 3 channel warna
    X_train_final = X_train_final.reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
    X_val = X_val.reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)

    print("Training Data:", X_train_final.shape, Y_train_final.shape)\n
    print("Validation Data:", X_val.shape, Y_val.shape)\n
    print("Testing Data:", X_test.shape, Y_test.shape)
    ''', language='python')

    st.header("Output:")
    st.markdown('''
    Training Data: (30038, 64, 64, 3) (30038, 7)\n
    Validation Data: (7510, 64, 64, 3) (7510, 7)\n
    Testing Data: (9387, 64, 64, 3) (9387, 7)
    ''')

    st.header("Model VGG19 Transfer Learning")
    st.code('''
    # Tentukan ukuran gambar yang digunakan (64x64 dan 3 channel warna)
    img_shape = (64, 64, 3)

    # Fungsi untuk membangun model
    def create_model():
        # Load VGG19 yang sudah dilatih sebelumnya tanpa top layers
        base_model = VGG19(include_top=False, input_shape=img_shape, weights="imagenet")

        # Freeze beberapa layer pada VGG19
        for layer in base_model.layers[:3]:  # Freeze 3 layer pertama
            layer.trainable = False

        # Menampilkan struktur model VGG19
        print("\n=== VGG19 Base Model Summary ===")
        base_model.summary()

        # Membangun model lengkap
        model = Sequential([
            base_model,
            Flatten(),
            BatchNormalization(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')  # 7 kelas output sesuai dengan dataset
        ])

        # Kompilasi model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossentropy(),
            metrics=[
                'accuracy',
                Recall(name='recall'),
                Precision(name='precision'),
                AUC(name='auc'),
                TruePositives(name='true_positives'),
                TrueNegatives(name='true_negatives'),
                FalseNegatives(name='false_negatives'),
                FalsePositives(name='false_positives')
            ]
        )
        return model

    # Memastikan apakah GPU tersedia
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

    # Membangun model dan menampilkan summary
    model = create_model()

    print("\n=== Full Model Summary ===")
    model.summary()
    ''', language='python')

    st.header("Output:")
    st.code("""
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │ input_layer (InputLayer)             │ (None, 64, 64, 3)           │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block1_conv1 (Conv2D)                │ (None, 64, 64, 64)          │           1,792 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block1_conv2 (Conv2D)                │ (None, 64, 64, 64)          │          36,928 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block1_pool (MaxPooling2D)           │ (None, 32, 32, 64)          │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block2_conv1 (Conv2D)                │ (None, 32, 32, 128)         │          73,856 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block2_conv2 (Conv2D)                │ (None, 32, 32, 128)         │         147,584 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block2_pool (MaxPooling2D)           │ (None, 16, 16, 128)         │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block3_conv1 (Conv2D)                │ (None, 16, 16, 256)         │         295,168 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block3_conv2 (Conv2D)                │ (None, 16, 16, 256)         │         590,080 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block3_conv3 (Conv2D)                │ (None, 16, 16, 256)         │         590,080 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block3_conv4 (Conv2D)                │ (None, 16, 16, 256)         │         590,080 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block3_pool (MaxPooling2D)           │ (None, 8, 8, 256)           │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block4_conv1 (Conv2D)                │ (None, 8, 8, 512)           │       1,180,160 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block4_conv2 (Conv2D)                │ (None, 8, 8, 512)           │       2,359,808 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block4_conv3 (Conv2D)                │ (None, 8, 8, 512)           │       2,359,808 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block4_conv4 (Conv2D)                │ (None, 8, 8, 512)           │       2,359,808 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block4_pool (MaxPooling2D)           │ (None, 4, 4, 512)           │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block5_conv1 (Conv2D)                │ (None, 4, 4, 512)           │       2,359,808 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block5_conv2 (Conv2D)                │ (None, 4, 4, 512)           │       2,359,808 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block5_conv3 (Conv2D)                │ (None, 4, 4, 512)           │       2,359,808 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block5_conv4 (Conv2D)                │ (None, 4, 4, 512)           │       2,359,808 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ block5_pool (MaxPooling2D)           │ (None, 2, 2, 512)           │               0 │
    └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
    Total params: 20,024,384 (76.39 MB)
    Trainable params: 19,985,664 (76.24 MB)
    Non-trainable params: 38,720 (151.25 KB)
    === Full Model Summary ===
    Model: "sequential"
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │ vgg19 (Functional)                   │ (None, 2, 2, 512)           │      20,024,384 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ flatten (Flatten)                    │ (None, 2048)                │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ batch_normalization                  │ (None, 2048)                │           8,192 │
    │ (BatchNormalization)                 │                             │                 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense (Dense)                        │ (None, 4096)                │       8,392,704 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dropout (Dropout)                    │ (None, 4096)                │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense_1 (Dense)                      │ (None, 4096)                │      16,781,312 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dropout_1 (Dropout)                  │ (None, 4096)                │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense_2 (Dense)                      │ (None, 7)                   │          28,679 │
    └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
    Total params: 45,235,271 (172.56 MB)
    Trainable params: 45,192,455 (172.40 MB)
    Non-trainable params: 42,816 (167.25 KB)
    """, language='text')

    st.header("Pelatihan Model")
    st.code('''
    # Konversi ke array numpy
    X_train_final = np.array(X_train_final)
    Y_train_final = np.array(Y_train_final, dtype=int)

    # 5-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    start_time_training = time.time()

    fold_no = 1
    val_accuracies = []
    best_val_accuracy = 0
    best_history = None
    best_fold = None
    histories = []

    for train_index, val_index in kf.split(X_train_final):
        print(f"### Training on Fold {fold_no} ###")

        # Split data
        X_train_fold, X_val_fold = X_train_final[train_index], X_train_final[val_index]
        Y_train_fold, Y_val_fold = Y_train_final[train_index], Y_train_final[val_index]

        # Normalisasi menggunakan preprocess_input
        X_train_fold = preprocess_input(X_train_fold)
        X_val_fold = preprocess_input(X_val_fold)

        # Callback: hanya ModelCheckpoint
        checkpoint = ModelCheckpoint(
            filepath=f'best_model_fold_{fold_no}.keras',
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            save_best_only=True
        )

        history = model.fit(
            X_train_fold,
            Y_train_fold,
            validation_data=(X_val_fold, Y_val_fold),
            epochs=30,
            batch_size=64,
            callbacks=[checkpoint],
            verbose=2
        )

        # Simpan hasil
        max_val_accuracy = np.max(history.history['val_accuracy'])
        val_accuracies.append(max_val_accuracy)
        histories.append(history.history)

        if max_val_accuracy > best_val_accuracy:
            best_val_accuracy = max_val_accuracy
            best_history = history
            best_fold = fold_no

        print(f"Akurasi validasi terbaik di Fold {fold_no}: {max_val_accuracy:.4f}")
        fold_no += 1

    # Akhiri timer
    end_time_training = time.time()
    training_time = end_time_training - start_time_training

    # Ringkasan hasil
    print(f"=== Hasil Akhir K-Fold ===")
    print(f"Akurasi validasi rata-rata: {np.mean(val_accuracies):.4f}")
    print(f"Standar deviasi akurasi: {np.std(val_accuracies):.4f}")
    print(f"Waktu total pelatihan: {training_time:.2f} detik")
    ''', language='python')

    st.header("Output:")
    st.code("""
    === Hasil Akhir K-Fold ===
    Akurasi validasi rata-rata: 0.9798
    Standar deviasi akurasi: 0.0239
    Waktu total pelatihan: 6663.85 detik
    """, language='text')

    st.header("Analisis Pelatihan Model")
    st.code('''
    # Cari akurasi dan loss validasi terbaik dari masing-masing fold
    best_val_accuracies = []
    best_val_losses = []

    # Untuk skor tambahan
    fold_precisions = []
    fold_recalls = []
    fold_aucs = []

    for i, history in enumerate(histories):
        best_accuracy = max(history['val_accuracy'])
        best_loss = min(history['val_loss'])

        best_val_accuracies.append(best_accuracy)
        best_val_losses.append(best_loss)

        print(f"Fold {i+1}: Best Val Accuracy = {best_accuracy:.4f}, Best Val Loss = {best_loss:.4f}")

        # === Tambahan: Evaluasi Precision, Recall, AUC dari model terbaik pada fold ini ===
        print(f"Evaluasi tambahan Fold {i+1}:")

        # Load model terbaik pada fold ke-(i+1)
        model = load_model(f'best_model_fold_{i+1}.keras')

        # Ambil kembali indeks validasi
        val_index = list(kf.split(X_train_final))[i][1]
        X_val_fold = preprocess_input(X_train_final[val_index])
        Y_val_fold = Y_train_final[val_index]

        # Prediksi
        y_pred = model.predict(X_val_fold)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(Y_val_fold, axis=1)

        # Hitung skor
        precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(Y_val_fold, y_pred, multi_class='ovr')
        except ValueError:
            auc = float('nan')  # Jika hanya satu kelas

        print(f"  Macro Precision: {precision:.4f}")
        print(f"  Macro Recall:    {recall:.4f}")
        print(f"  Macro AUC:       {auc:.4f}")

        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_aucs.append(auc)

    # Rata-rata akurasi dan loss validasi terbaik
    mean_best_accuracy = np.mean(best_val_accuracies)
    mean_best_loss = np.mean(best_val_losses)

    print(f"Average Best Validation Accuracy Across Folds: {mean_best_accuracy:.4f}")
    print(f"Average Best Validation Loss Across Folds: {mean_best_loss:.4f}")

    # Tentukan fold terbaik berdasarkan akurasi tertinggi
    best_acc_fold = np.argmax(best_val_accuracies) + 1
    print(f"Best Accuracy Model is from Fold {best_acc_fold} with Val Accuracy: {best_val_accuracies[best_acc_fold-1]:.4f}")

    # Tentukan fold terbaik berdasarkan loss terendah
    best_loss_fold = np.argmin(best_val_losses) + 1
    print(f"Best Loss Model is from Fold {best_loss_fold} with Val Loss: {best_val_losses[best_loss_fold-1]:.4f}")

    # Tambahan: Rata-rata skor evaluasi
    print(f"=== Rata-rata Evaluasi Tambahan Across Folds ===")
    print(f"Average Macro Precision: {np.mean(fold_precisions):.4f}")
    print(f"Average Macro Recall:    {np.mean(fold_recalls):.4f}")
    print(f"Average Macro AUC:       {np.nanmean(fold_aucs):.4f}")
    ''', language='python')

    st.header("Output:")
    st.code("""
    Fold 1: Best Val Accuracy = 0.9349, Best Val Loss = 0.2491
    Evaluasi tambahan Fold 1:
    188/188 ━━━━━━━━━━━━━━━━━━━━ 8s 28ms/step
      Macro Precision: 0.9329
      Macro Recall:    0.9332
      Macro AUC:       0.9908
    Fold 2: Best Val Accuracy = 0.9752, Best Val Loss = 0.0835
    Evaluasi tambahan Fold 2:
    188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 19ms/step
      Macro Precision: 0.9755
      Macro Recall:    0.9756
      Macro AUC:       0.9984
    Fold 3: Best Val Accuracy = 0.9938, Best Val Loss = 0.0375
    Evaluasi tambahan Fold 3:
    188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 19ms/step
      Macro Precision: 0.9939
      Macro Recall:    0.9937
      Macro AUC:       0.9997
    Fold 4: Best Val Accuracy = 0.9973, Best Val Loss = 0.0143
    Evaluasi tambahan Fold 4:
    188/188 ━━━━━━━━━━━━━━━━━━━━ 6s 27ms/step
      Macro Precision: 0.9974
      Macro Recall:    0.9974
      Macro AUC:       0.9997
    Fold 5: Best Val Accuracy = 0.9978, Best Val Loss = 0.0106
    Evaluasi tambahan Fold 5:
    188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 19ms/step
      Macro Precision: 0.9978
      Macro Recall:    0.9978
      Macro AUC:       0.9998

    Average Best Validation Accuracy Across Folds: 0.9798
    Average Best Validation Loss Across Folds: 0.0790
    Best Accuracy Model is from Fold 5 with Val Accuracy: 0.9978
    Best Loss Model is from Fold 5 with Val Loss: 0.0106

    === Rata-rata Evaluasi Tambahan Across Folds ===
    Average Macro Precision: 0.9795
    Average Macro Recall:    0.9795
    Average Macro AUC:       0.9977
    """, language='text')


    st.header("ROC Hasil Pelatihan Model")
    st.code('''
    # Plot semua Fold - Gabungan Akurasi Validasi
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history['val_accuracy'], label=f'Fold {i+1} Validation Accuracy')

    plt.title('Validation Accuracy Across All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot semua Fold - Gabungan Loss
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history['val_loss'], label=f'Fold {i+1} Validation Loss')

    plt.title('Validation Loss Across All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    ''', language='python')
    st.header("Output:")
    image_path = 'roc 1.png'
    st.image(Image.open(image_path), caption='Validasi Akurasi Setiap Fold', width=800)

    image_path = 'roc 2.png'
    st.image(Image.open(image_path), caption='Validasi Loss Setiap Fold', width=800)


    st.header("Simpan Model Terbaik")
    st.code("""
    metrics_data = {
    'epoch': list(range(1, len(best_history.history['accuracy']) + 1)),
    'train_accuracy': best_history.history['accuracy'],
    'val_accuracy': best_history.history['val_accuracy'],
    'train_loss': best_history.history['loss'],
    'val_loss': best_history.history['val_loss'],
    }

    metrics_df = pd.DataFrame(metrics_data)
    csv_path = f'training_metrics_fold_{best_fold}.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrik pelatihan dari Fold {best_fold} berhasil disimpan ke '{csv_path}'")

    model_best = f'best_model_fold_{best_fold}.keras'
    print(f"Model terbaik disimpan sebagai: {model_best}")
    """)

    st.header("Model Pengujian")
    st.code("""
    # Preprocessing data testing
    X_test_preprocessed = preprocess_input(X_test)

    # Load model terbaik dari fold terbaik
    best_model_path = f'best_model_fold_{best_fold}.keras'
    best_model = load_model(best_model_path)

    # Evaluasi model
    results = best_model.evaluate(X_test_preprocessed, Y_test, verbose=2)
    print("\n=== Hasil Evaluasi ===")
    for name, value in zip(best_model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    # Prediksi
    y_pred_probs = best_model.predict(X_test_preprocessed)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(Y_test, axis=1)

    # Laporan klasifikasi
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    """)

    st.header("Output:")
    st.code("""
    294/294 - 10s - 36ms/step - accuracy: 0.9705 - auc: 0.9942 - false_negatives: 293.0000 - false_positives: 261.0000 - loss: 0.1603 - precision: 0.9721 - recall: 0.9688 - true_negatives: 56061.0000 - true_positives: 9094.0000

    === Hasil Evaluasi ===
    loss: 0.1603
    compile_metrics: 0.9705
    294/294 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step

    === Classification Report ===
                  precision    recall  f1-score   support

              0       0.99      1.00      1.00      1341
              1       0.99      1.00      1.00      1341
              2       0.93      0.99      0.96      1341
              3       1.00      1.00      1.00      1341
              4       0.99      0.82      0.89      1341
              5       0.91      0.99      0.95      1341
              6       1.00      1.00      1.00      1341

        accuracy                           0.97      9387
      macro avg       0.97      0.97      0.97      9387
    weighted avg       0.97      0.97      0.97      9387
    """, language='text')
    image_path = 'confusion.png'
    st.image(Image.open(image_path), caption='Confusion Matrix', width=800)

    st.code("""
    # Tentukan jumlah kelas dari Y_test
    n_classes = Y_test.shape[1]

    # Ambil fold terbaik berdasarkan akurasi (sudah dihitung sebelumnya)
    best_model_path = f'best_model_fold_{best_acc_fold}.keras'
    best_model = load_model(best_model_path)

    # Preprocessing data testing
    X_test_preprocessed = preprocess_input(X_test)

    # Prediksi probabilitas kelas dari model terbaik
    y_pred_probs = best_model.predict(X_test_preprocessed)

    # Konversi label Y_test menjadi label integer
    y_true = np.argmax(Y_test, axis=1)

    # Binarisasi label untuk keperluan ROC per kelas
    y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

    # Inisialisasi variabel untuk menyimpan nilai fpr, tpr, dan AUC per kelas
    fpr = dict()
    tpr = dict()
    roc_auc_scores = dict()  # Hindari nama 'auc'

    # Hitung ROC Curve dan AUC untuk setiap kelas
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_probs[:, i])
        roc_auc_scores[i] = sklearn_auc(fpr[i], tpr[i])

    # Visualisasi ROC Curve
    plt.figure(figsize=(12, 8))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc_scores[i]:.2f})')

    # Garis acuan diagonal (random guess)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    # Pengaturan tampilan
    plt.title(f'ROC Curve for Each Class - Best Model from Fold {best_acc_fold}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    # Tampilkan plot
    plt.show()
    """)
    st.header("Output:")
    image_path = 'roc.png'
    st.image(Image.open(image_path), caption='Validasi Akurasi Setiap Fold', width=800)

# --- 3. Prediksi Gambar ---
elif menu == "Prediksi Gambar":
    st.title("🩺 Klasifikasi Kanker Kulit dengan VGG19")
    st.markdown("Upload gambar kulit berformat **.jpg / .jpeg / .png** untuk prediksi.")

    class_labels = [
        'actinic keratoses and intraepithelial carcinomae',
        'basal cell carcinoma',
        'benign keratosis-like lesions',
        'dermatofibroma',
        'melanocytic nevi',
        'melanoma',
        'vascular lesions',
    ]

    # PATH ke model ONNX (pastikan file ini sudah ada di repo dan di-push)
    onnx_model_path = "best_model_fold_5.onnx"

    # coba load session ONNX
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        st.error(f"Model ONNX tidak ditemukan atau gagal dimuat: {e}")
        st.info("Pastikan file 'best_model_fold_5.onnx' berada di folder project dan sudah di-push ke repository.")
        st.stop()

    uploaded_file = st.file_uploader("📁 Unggah Gambar", type=["jpg", "jpeg", "png"])

    def preprocess_for_64(img: Image.Image):
        """
        Preprocessing untuk model yang menerima 64x64 RGB normalized (0-1).
        Jika modelmu memakai preprocessing lain (contoh: mean subtraction), sesuaikan di sini.
        """
        img = img.resize((64, 64))
        arr = np.array(img).astype(np.float32) / 255.0  # normalisasi 0-1
        # Pastikan input shape = (1, 64, 64, 3)
        arr = np.expand_dims(arr, axis=0)
        return arr

    if uploaded_file is not None:
        try:
            # Baca gambar dan tampilkan
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="🖼 Gambar yang Diupload", use_column_width=True)

            # Preprocess sesuai model (64x64 di asumsi ini)
            input_arr = preprocess_for_64(img).astype(np.float32)

            # Dapatkan nama input ONNX
            input_name = ort_session.get_inputs()[0].name

            # Jalankan inference
            outputs = ort_session.run(None, {input_name: input_arr})
            prediction = np.array(outputs[0])

            # Pastikan bentuk (N,) untuk proba
            if prediction.ndim == 2:
                prediction = prediction[0]

            # Ambil kelas dan confidence
            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[predicted_class])

            # Tampilkan hasil
            st.header("📌 Hasil Prediksi:")
            st.write(f"**Kelas:** {class_labels[predicted_class]}")
            st.write(f"**Tingkat Keyakinan:** {confidence * 100:.2f}%")

            # DataFrame probabilitas (urut sesuai class_labels)
            df = pd.DataFrame({
                "Kelas": class_labels,
                "Probabilitas": prediction
            })

            df_sorted = df.sort_values("Probabilitas", ascending=False)

            st.header("📊 Probabilitas Tiap Kelas:")
            chart = alt.Chart(df_sorted).mark_bar().encode(
                x=alt.X("Kelas:N", sort=df_sorted["Kelas"].tolist(), axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Probabilitas:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("Kelas:N"), alt.Tooltip("Probabilitas:Q", format=".4f")]
            ).properties(width=700, height=400)

            st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
