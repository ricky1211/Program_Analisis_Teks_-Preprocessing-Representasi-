# --- (BAGIAN 0) IMPORT SEMUA LIBRARY ---
import re
import pandas as pd
import numpy as np
import os
import tempfile

# 0.1 Library Web Framework
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
import speech_recognition as sr
import requests
from bs4 import BeautifulSoup

# --- Konfigurasi Awal ---
@st.cache_resource
def inisialisasi_model():
    """Menginisialisasi model Sastrawi"""
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    return stemmer, stopword_remover

# Panggil fungsi inisialisasi
try:
    stemmer, stopword_remover = inisialisasi_model()
    st.success("✅ Model Sastrawi berhasil dimuat", icon="✅")
except Exception as e:
    st.error(f"❌ Gagal memuat model Sastrawi: {e}")
    st.stop()


# --- (BAGIAN 1) FUNGSI PREPROCESSING ---

def tokenize_manual(text):
    """
    Tokenisasi manual menggunakan regex.
    Lebih stabil daripada NLTK word_tokenize.
    """
    # Pisahkan kata berdasarkan huruf dan angka
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    return tokens

def preprocess_text(text):
    """Fungsi preprocessing teks tanpa NLTK."""
    try:
        st.info("   📝 Case folding & cleaning...")
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        st.info("   ✂️ Tokenizing...")
        tokens = tokenize_manual(text)
        
        if not tokens:
            st.warning("⚠️ Tidak ada token yang dihasilkan")
            return ""
        
        st.info("   🚫 Stopword removal...")
        text_tanpa_stopword = stopword_remover.remove(' '.join(tokens))
        
        st.info("   🌱 Stemming...")
        tokens_tanpa_stopword = text_tanpa_stopword.split()
        stemmed_tokens = [stemmer.stem(token) for token in tokens_tanpa_stopword]
        
        hasil = ' '.join(stemmed_tokens)
        st.success(f"   ✅ Preprocessing selesai! Token: {len(stemmed_tokens)}")
        return hasil
        
    except Exception as e:
        st.error(f"❌ Error saat preprocessing: {e}")
        return ""


# --- (BAGIAN 2) FUNGSI ANALISIS ---

def run_analysis(list_dokumen_bersih):
    """Menganalisis dokumen dengan BoW, TF-IDF, dan Word2Vec"""
    st.header("📊 IMPLEMENTASI REPRESENTASI TEKS")

    # --- METODE 1: BAG OF WORDS ---
    st.subheader("1️⃣ Metode Bag of Words (BoW)")
    try:
        bow_vectorizer = CountVectorizer()
        bow_matrix = bow_vectorizer.fit_transform(list_dokumen_bersih)
        fitur_bow = bow_vectorizer.get_feature_names_out()
        
        df_bow = pd.DataFrame(
            bow_matrix.toarray(), 
            columns=fitur_bow, 
            index=[f'Dokumen {i+1}' for i in range(len(list_dokumen_bersih))]
        )
        
        st.dataframe(df_bow, use_container_width=True)
        st.info("💡 **Penjelasan:** Menghitung frekuensi kemunculan setiap kata dalam dokumen.")
        
        # Statistik
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Kata Unik", len(fitur_bow))
        with col2:
            st.metric("Total Kata (Sum)", int(bow_matrix.sum()))
            
    except ValueError as e:
        st.error(f"❌ ERROR BoW: {e}")

    # --- METODE 2: TF-IDF ---
    st.subheader("2️⃣ Metode TF-IDF")
    try:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(list_dokumen_bersih)
        fitur_tfidf = tfidf_vectorizer.get_feature_names_out()
        
        df_tfidf = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=fitur_tfidf, 
            index=[f'Dokumen {i+1}' for i in range(len(list_dokumen_bersih))]
        )
        
        st.dataframe(df_tfidf.round(4), use_container_width=True)
        st.info("💡 **Penjelasan:** Memberi bobot kata berdasarkan frekuensi dan kelangkaannya di seluruh dokumen.")
        
        # Kata dengan TF-IDF tertinggi
        max_tfidf = df_tfidf.max().sort_values(ascending=False).head(5)
        st.write("**Top 5 Kata Berdasarkan TF-IDF:**")
        st.bar_chart(max_tfidf)
        
    except ValueError as e:
        st.error(f"❌ ERROR TF-IDF: {e}")

    # --- METODE 3: WORD2VEC ---
    st.subheader("3️⃣ Metode Word2Vec")
    try:
        tokenized_docs_w2v = [doc.split() for doc in list_dokumen_bersih if doc]
        
        if not tokenized_docs_w2v:
            st.error("❌ ERROR: Tidak ada token untuk dilatih.")
            return

        st.write("**Input untuk Word2Vec (Sample):**")
        st.code(str(tokenized_docs_w2v[:2]), language="python")
        
        # Training Word2Vec
        with st.spinner("🔄 Melatih model Word2Vec..."):
            model_w2v = Word2Vec(
                sentences=tokenized_docs_w2v, 
                vector_size=100, 
                window=5, 
                min_count=1, 
                workers=4,
                epochs=10
            )
        
        st.success("✅ Model Word2Vec berhasil dilatih!")
        
        # Vocabulary info
        vocab_size = len(model_w2v.wv.index_to_key)
        st.metric("Ukuran Vocabulary", vocab_size)
        
        if model_w2v.wv.index_to_key:
            kata_uji = model_w2v.wv.index_to_key[0]
            
            # Tampilkan vektor
            st.write(f"**Contoh Vektor untuk kata '{kata_uji}'** (10 dimensi pertama):")
            vektor = model_w2v.wv[kata_uji][:10].tolist()
            st.json({f"dim_{i}": round(v, 4) for i, v in enumerate(vektor)})

            # Kata mirip
            try:
                kata_mirip = model_w2v.wv.most_similar(kata_uji, topn=5)
                st.write(f"**Kata yang paling mirip dengan '{kata_uji}':**")
                
                df_mirip = pd.DataFrame(kata_mirip, columns=['Kata', 'Similarity Score'])
                st.dataframe(df_mirip, use_container_width=True)
                
            except KeyError:
                st.warning(f"⚠️ Kata '{kata_uji}' tidak memiliki kata mirip.")
        else:
            st.error("❌ Vocabulary kosong.")
            
    except Exception as e:
        st.error(f"❌ ERROR Word2Vec: {e}")


# --- (BAGIAN 3) FUNGSI HANDLER INPUT ---

def handle_url(url):
    """Ekstrak teks dari URL artikel menggunakan BeautifulSoup"""
    st.info(f"🌐 Mengunduh artikel dari: {url}")
    try:
        # Set headers untuk menghindari blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download halaman
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse dengan BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Hapus script dan style
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Ekstrak teks dari paragraf
        paragraphs = soup.find_all(['p', 'article', 'div'])
        text_list = []
        
        for para in paragraphs:
            text = para.get_text(strip=True)
            if len(text) > 50:  # Filter teks yang terlalu pendek
                text_list.append(text)
        
        full_text = '\n'.join(text_list)
        
        if len(full_text) < 100:
            st.warning("⚠️ Teks yang diekstrak terlalu pendek. Coba URL lain.")
            return ""
        
        return full_text
        
    except requests.exceptions.Timeout:
        st.error("❌ Timeout: Server terlalu lama merespons.")
        return ""
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Gagal mengambil artikel: {e}")
        return ""
    except Exception as e:
        st.error(f"❌ Error parsing artikel: {e}")
        return ""

def handle_audio(file_path):
    """Transkripsi audio ke teks"""
    st.info(f"🎤 Mentranskripsi file audio...")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="id-ID")
            return text
    except Exception as e:
        st.error(f"❌ Gagal transkripsi audio: {e}")
        return ""

def handle_doc(file_path, file_extension):
    """Ekstrak teks dari dokumen (.txt, .pdf, .docx)"""
    text = ""
    try:
        if file_extension == '.txt':
            st.info("📄 Membaca file TXT...")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
        elif file_extension == '.pdf':
            st.info("📕 Membaca file PDF...")
            with pdfplumber.open(file_path) as pdf:
                all_pages_text = [p.extract_text() for p in pdf.pages if p.extract_text()]
                text = "\n".join(all_pages_text)
                
        elif file_extension == '.docx':
            st.info("📘 Membaca file DOCX...")
            doc = docx.Document(file_path)
            all_paras_text = [para.text for para in doc.paragraphs]
            text = "\n".join(all_paras_text)
            
    except Exception as e:
        st.error(f"❌ Error membaca file: {e}")
        
    return text

def save_uploaded_file(uploaded_file):
    """Simpan file upload ke temporary directory"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"❌ Gagal menyimpan file: {e}")
        return None


# --- (BAGIAN 4) APLIKASI UTAMA ---

# Inisialisasi session state
if 'dokumen_mentah_list' not in st.session_state:
    st.session_state.dokumen_mentah_list = []

# --- UI HEADER ---
st.title("🚀 Program Analisis Teks NLP")
st.markdown("""
**Aplikasi ini melakukan:**
- 🔄 Preprocessing (Case Folding, Tokenizing, Stopword Removal, Stemming)
- 📊 Representasi Teks (Bag of Words, TF-IDF, Word2Vec)
- 📁 Mendukung berbagai input: URL, PDF, DOCX, TXT, Audio
""")

st.divider()

# --- SIDEBAR: INPUT DATA ---
st.sidebar.header("📥 Input Data")

# INPUT 1: URL
with st.sidebar.expander("🌐 Dari URL"):
    url_input = st.text_input("Masukkan URL artikel")
    if st.button("Proses URL", key="btn_url"):
        if url_input:
            with st.spinner("Memproses URL..."):
                teks_mentah = handle_url(url_input)
                if teks_mentah:
                    st.session_state.dokumen_mentah_list.append(teks_mentah)
                    st.success(f"✅ URL berhasil diproses! Total dokumen: {len(st.session_state.dokumen_mentah_list)}")
        else:
            st.warning("⚠️ Harap masukkan URL.")

# INPUT 2: FILE UPLOAD
with st.sidebar.expander("📁 Dari File"):
    uploaded_file = st.file_uploader(
        "Pilih file", 
        type=["wav", "mp3", "pdf", "docx", "txt"],
        help="Format yang didukung: Audio (WAV, MP3), Dokumen (PDF, DOCX, TXT)"
    )
    
    if uploaded_file is not None:
        st.info(f"📄 File: {uploaded_file.name}")
        
        if st.button("Proses File", key="btn_file"):
            with st.spinner(f"Memproses {uploaded_file.name}..."):
                file_path = save_uploaded_file(uploaded_file)
                
                if file_path:
                    _, file_extension = os.path.splitext(uploaded_file.name)
                    file_extension = file_extension.lower()
                    
                    teks_mentah = ""
                    
                    # Pilih handler
                    if file_extension in ['.wav', '.mp3']:
                        teks_mentah = handle_audio(file_path)
                    elif file_extension in ['.txt', '.pdf', '.docx']:
                        teks_mentah = handle_doc(file_path, file_extension)
                    else:
                        st.error(f"❌ Format '{file_extension}' tidak didukung.")
                    
                    # Hapus file temporary
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    
                    # Tambahkan ke list jika berhasil
                    if teks_mentah:
                        st.session_state.dokumen_mentah_list.append(teks_mentah)
                        st.success(f"✅ File berhasil diekstrak! Total dokumen: {len(st.session_state.dokumen_mentah_list)}")
                        
                        with st.expander("👁️ Lihat teks hasil ekstraksi"):
                            st.text_area("", teks_mentah, height=150, key="preview_text")

# Tombol reset
if st.sidebar.button("🗑️ Hapus Semua Dokumen", type="secondary"):
    st.session_state.dokumen_mentah_list = []
    st.rerun()

# Status dokumen
st.sidebar.divider()
st.sidebar.metric("📚 Total Dokumen", len(st.session_state.dokumen_mentah_list))

# --- MAIN CONTENT ---

# Tampilkan dokumen yang terkumpul
if st.button("📋 Tampilkan Dokumen Terkumpul"):
    if not st.session_state.dokumen_mentah_list:
        st.warning("⚠️ Belum ada dokumen yang dikumpulkan.")
    else:
        st.subheader(f"📚 Total {len(st.session_state.dokumen_mentah_list)} Dokumen")
        
        for i, doc in enumerate(st.session_state.dokumen_mentah_list):
            with st.expander(f"📄 Dokumen {i+1} ({len(doc)} karakter)"):
                st.text_area("", doc, height=150, key=f"doc_raw_{i}", disabled=True)

st.divider()

# Tombol untuk analisis
st.header("🎯 Jalankan Analisis")

if st.button("▶️ MULAI PREPROCESSING & ANALISIS", type="primary", use_container_width=True):
    if not st.session_state.dokumen_mentah_list:
        st.error("❌ Tidak ada dokumen untuk dianalisis. Silakan tambahkan dokumen terlebih dahulu.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.subheader("🔄 PROSES PREPROCESSING")
        list_dokumen_bersih = []
        
        total_docs = len(st.session_state.dokumen_mentah_list)
        
        for i, doc_mentah in enumerate(st.session_state.dokumen_mentah_list):
            status_text.text(f"Memproses dokumen {i+1}/{total_docs}...")
            progress_bar.progress((i + 1) / total_docs)
            
            with st.expander(f"📝 Preprocessing Dokumen {i+1}", expanded=(i==0)):
                hasil_proses = preprocess_text(doc_mentah)
                
                if hasil_proses:
                    list_dokumen_bersih.append(hasil_proses)
                    st.text_area(
                        f"Hasil Bersih (Dokumen {i+1})", 
                        hasil_proses, 
                        height=100, 
                        key=f"clean_doc_{i}"
                    )
                else:
                    st.error(f"❌ Dokumen {i+1} gagal diproses.")
        
        progress_bar.empty()
        status_text.empty()
        
        if not list_dokumen_bersih:
            st.error("❌ Semua dokumen gagal diproses. Tidak ada yang bisa dianalisis.")
        else:
            st.success(f"✅ Preprocessing selesai! {len(list_dokumen_bersih)} dokumen berhasil diproses.")
            
            st.divider()
            
            # Jalankan analisis
            run_analysis(list_dokumen_bersih)
            
            st.balloons()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <small>📚 Program Analisis Teks NLP | Dibuat dengan Streamlit</small>
</div>
""", unsafe_allow_html=True)
