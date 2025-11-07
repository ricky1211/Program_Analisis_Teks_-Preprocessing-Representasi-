# --- (BAGIAN 0) IMPORT SEMUA LIBRARY ---
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import os
import tempfile  # Diperlukan untuk menyimpan file sementara

# 0.1 Library Web Framework (BARU)
import streamlit as st

# 0.2 Library Preprocessing (Sastrawi)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 0.3 Library Representasi Teks
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# 0.4 Library Handler Input
import docx
import pdfplumber
from newspaper import Article
import speech_recognition as sr
# Baris 'from moviepy.editor' TELAH DIHAPUS

# --- Konfigurasi Awal (Hanya dijalankan sekali) ---
@st.cache_resource  # Streamlit akan 'mengingat' objek ini
def inisialisasi_model():
    # Download NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    
    # Inisialisasi Sastrawi
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    return stemmer, stopword_remover

# Panggil fungsi inisialisasi
stemmer, stopword_remover = inisialisasi_model()


# --- (BAGIAN 1) FUNGSI PREPROCESSING (PROYEK 1) ---

def preprocess_text(text):
    """Fungsi preprocessing teks."""
    st.info("   ...melakukan case folding & cleaning...")
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    st.info("   ...melakukan tokenizing...")
    tokens = word_tokenize(text)
    
    st.info("   ...melakukan stopword removal...")
    text_tanpa_stopword = stopword_remover.remove(' '.join(tokens))
    
    st.info("   ...melakukan stemming...")
    tokens_tanpa_stopword = text_tanpa_stopword.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens_tanpa_stopword]
    
    return ' '.join(stemmed_tokens)

# --- (BAGIAN 2) FUNGSI ANALISIS (PROYEK 2) ---

def run_analysis(list_dokumen_bersih):
    """
    Mengambil list dokumen bersih dan menampilkan
    hasil analisis BoW, TF-IDF, dan Word2Vec di web.
    """
    st.header("--- 2. IMPLEMENTASI REPRESENTASI TEKS ---")

    # --- METODE 1: BAG OF WORDS (BoW) ---
    st.subheader("2.1. Metode Bag of Words (BoW)")
    try:
        bow_vectorizer = CountVectorizer()
        bow_matrix = bow_vectorizer.fit_transform(list_dokumen_bersih)
        fitur_bow = bow_vectorizer.get_feature_names_out()
        df_bow = pd.DataFrame(bow_matrix.toarray(), columns=fitur_bow, index=[f'Dok {i+1}' for i in range(len(list_dokumen_bersih))])
        st.dataframe(df_bow) # Menggunakan st.dataframe untuk tabel interaktif
        st.write("Penjelasan: Menghitung frekuensi kemunculan setiap kata.")
    except ValueError:
        st.error("ERROR BoW: Teks tidak mengandung kata yang bisa di-vektorisasi.")

    # --- METODE 2: TF-IDF ---
    st.subheader("2.2. Metode TF-IDF")
    try:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(list_dokumen_bersih)
        fitur_tfidf = tfidf_vectorizer.get_feature_names_out()
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=fitur_tfidf, index=[f'Dok {i+1}' for i in range(len(list_dokumen_bersih))])
        st.dataframe(df_tfidf.round(2))
        st.write("Penjelasan: Memberi bobot kata berdasarkan frekuensi dan kelangkaannya.")
    except ValueError:
        st.error("ERROR TF-IDF: Teks tidak mengandung kata yang bisa di-vektorisasi.")

    # --- METODE 3: WORD2VEC ---
    st.subheader("2.3. Metode Word2Vec")
    tokenized_docs_w2v = [doc.split() for doc in list_dokumen_bersih if doc]
    if not tokenized_docs_w2v:
        st.error("ERROR Word2Vec: Tidak ada token untuk dilatih.")
        return

    st.write("Input untuk Word2Vec (List of List Tokens):")
    st.json(tokenized_docs_w2v) # st.json untuk menampilkan list
    
    model_w2v = Word2Vec(sentences=tokenized_docs_w2v, vector_size=100, window=5, min_count=1, workers=4)
    st.success("Model Word2Vec berhasil dilatih.")
    
    if model_w2v.wv.index_to_key:
        kata_uji = model_w2v.wv.index_to_key[0] # Ambil kata pertama
        try:
            st.write(f"Contoh representasi vektor untuk kata '{kata_uji}':")
            st.json(model_w2v.wv[kata_uji][:10].tolist()) # Tampilkan 10 dimensi

            kata_mirip = model_w2v.wv.most_similar(kata_uji, topn=3)
            st.write(f"Kata yang paling mirip dengan '{kata_uji}':")
            st.json(kata_mirip)
        except KeyError:
            st.error(f"Kata '{kata_uji}' tidak ada dalam vocabulary.")
    else:
        st.error("Tidak ada vocabulary yang terbentuk di Word2Vec.")

# --- (BAGIAN 3) FUNGSI HANDLER INPUT ---

def handle_url(url):
    st.info(f"Mengunduh artikel dari: {url}")
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Gagal mengambil artikel dari URL: {e}")
        return ""

def handle_audio(file_path):
    st.info(f"Mentranskripsi file audio: {file_path}")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="id-ID")
            return text
    except Exception as e:
        st.error(f"Gagal transkripsi audio: {e}. (Perlu koneksi internet)")
        return ""

# FUNGSI handle_video() TELAH DIHAPUS

def handle_doc(file_path, file_extension):
    """
    Fungsi gabungan untuk menangani file dokumen
    (.txt, .pdf, .docx)
    """
    text = ""
    if file_extension == '.txt':
        st.info("Handler dipilih: Teks (.txt)")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif file_extension == '.pdf':
        st.info("Handler dipilih: PDF (.pdf)")
        with pdfplumber.open(file_path) as pdf:
            all_pages_text = [p.extract_text() for p in pdf.pages if p.extract_text()]
            text = "\n".join(all_pages_text)
    elif file_extension == '.docx':
        st.info("Handler dipilih: Word (.docx)")
        doc = docx.Document(file_path)
        all_paras_text = [para.text for para in doc.paragraphs]
        text = "\n".join(all_paras_text)
    return text

def save_uploaded_file(uploaded_file):
    """
    Menyimpan file yang di-upload Streamlit ke disk sementara
    dan mengembalikan path-nya.
    """
    try:
        # Buat temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            # Tulis konten file yang diupload ke file sementara
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name  # Kembalikan path ke file sementara
    except Exception as e:
        st.error(f"Gagal menyimpan file sementara: {e}")
        return None

# --- (BAGIAN 4) PROSES UTAMA / APLIKASI WEB ---

# Inisialisasi 'session state' untuk menyimpan daftar dokumen
if 'dokumen_mentah_list' not in st.session_state:
    st.session_state.dokumen_mentah_list = []

# --- Tampilan UI (User Interface) ---

st.title("🚀 Program Analisis Teks (Preprocessing & Representasi)")
st.write("Aplikasi ini mengubah teks dari berbagai sumber (file, URL, audio) menjadi data numerik (BoW, TF-IDF, Word2Vec).")

# --- Sidebar untuk Input ---
st.sidebar.header("Input Data")

# Input 1: URL
url_input = st.sidebar.text_input("Masukkan URL Artikel")
if st.sidebar.button("Proses URL"):
    if url_input:
        with st.spinner("Memproses URL..."):
            teks_mentah = handle_url(url_input)
            if teks_mentah:
                st.session_state.dokumen_mentah_list.append(teks_mentah)
                st.sidebar.success("URL berhasil diproses dan teks ditambahkan!")
    else:
        st.sidebar.warning("Harap masukkan URL.")

# Input 2: File Upload
st.sidebar.subheader("Atau Upload File")
# Tipe file video (mp4, mkv, mov) TELAH DIHAPUS dari list di bawah
uploaded_file = st.sidebar.file_uploader("Pilih file (audio, .pdf, .docx, .txt)", type=["wav", "mp3", "pdf", "docx", "txt"])

if uploaded_file is not None:
    # Tampilkan info file
    st.sidebar.info(f"File diterima: {uploaded_file.name}")
    
    # Tombol untuk memproses file
    if st.sidebar.button("Proses File"):
        with st.spinner(f"Memproses {uploaded_file.name}..."):
            # Simpan file ke disk sementara
            file_path = save_uploaded_file(uploaded_file)
            
            if file_path:
                _, file_extension = os.path.splitext(uploaded_file.name)
                file_extension = file_extension.lower()
                
                teks_mentah = ""
                # Pilih handler berdasarkan ekstensi
                if file_extension in ['.wav', '.mp3', '.flac']:
                    teks_mentah = handle_audio(file_path)
                # ELIF untuk video TELAH DIHAPUS
                elif file_extension in ['.txt', '.pdf', '.docx']:
                    teks_mentah = handle_doc(file_path, file_extension)
                else:
                    st.error(f"Ekstensi file '{file_extension}' tidak didukung.")
                
                # Hapus file sementara setelah diproses
                os.remove(file_path)
                
                # Jika berhasil, tambahkan ke list
                if teks_mentah:
                    st.session_state.dokumen_mentah_list.append(teks_mentah)
                    st.sidebar.success("File berhasil diproses dan teks ditambahkan!")
                    st.subheader("Teks Mentah Hasil Ekstraksi:")
                    st.text_area("", teks_mentah, height=150)

# --- Kontrol dan Tampilan Halaman Utama ---

st.divider()

# Tombol untuk menampilkan dokumen yang terkumpul
if st.button("Tampilkan Teks Mentah yang Terkumpul"):
    if not st.session_state.dokumen_mentah_list:
        st.warning("Belum ada teks yang dikumpulkan.")
    else:
        st.subheader(f"Total {len(st.session_state.dokumen_mentah_list)} Dokumen Terkumpul")
        for i, doc in enumerate(st.session_state.dokumen_mentah_list):
            st.text_area(f"Dokumen {i+1}", doc, height=100, key=f"doc_{i}")

# Tombol untuk menjalankan analisis
st.header("Jalankan Analisis")
if st.button("MULAI PREPROCESSING & ANALISIS", type="primary"):
    if not st.session_state.dokumen_mentah_list:
        st.error("Tidak ada dokumen untuk dianalisis. Silakan tambahkan file atau URL.")
    else:
        with st.spinner("Analisis sedang berjalan... Ini mungkin butuh waktu lama..."):
            st.subheader("--- 1. PROSES PREPROCESSING DIMULAI ---")
            list_dokumen_bersih = []
            for i, doc_mentah in enumerate(st.session_state.dokumen_mentah_list):
                st.write(f"Memproses Dokumen {i+1}...")
                hasil_proses = preprocess_text(doc_mentah)
                list_dokumen_bersih.append(hasil_proses)
                st.text_area(f"Dokumen {i+1} (Bersih)", hasil_proses, height=100, key=f"clean_doc_{i}")
            
            st.success("--- PROSES PREPROCESSING SELESAI ---")
            
            # Panggil fungsi analisis
            run_analysis(list_dokumen_bersih)

# Tombol untuk mereset
if st.sidebar.button("Bersihkan Semua Dokumen"):
    st.session_state.dokumen_mentah_list = []
    st.experimental_rerun() # Ganti dengan st.rerun() jika versi Streamlit Anda baru