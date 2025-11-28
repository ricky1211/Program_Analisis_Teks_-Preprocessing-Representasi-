# --- IMPORT LIBRARY ---
import re
import pandas as pd
import numpy as np
import os
import tempfile
import streamlit as st
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Preprocessing
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Representasi Teks
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Document Processing
import docx
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, SimpleRNN
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    
# NLP Advanced
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk.chunk import ne_chunk


# --- DOWNLOAD NLTK DATA ---
@st.cache_resource
def download_nltk_data():
    """Download semua data NLTK yang diperlukan"""
    nltk_packages = [
        'punkt',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    for package in nltk_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass
        
        try:
            nltk.data.find(f'taggers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass
        
        try:
            nltk.data.find(f'chunkers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass
        
        try:
            nltk.data.find(f'corpora/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass

try:
    download_nltk_data()
except Exception as e:
    st.warning(f"⚠️ Beberapa data NLTK gagal didownload atau dimuat: {e}")


# --- INISIALISASI MODEL SASTRAWI ---
@st.cache_resource
def inisialisasi_model():
    """Menginisialisasi model Sastrawi (Stemmer & Stopword Remover)"""
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    return stemmer, stopword_remover

try:
    stemmer, stopword_remover = inisialisasi_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model Sastrawi: {e}")
    pass 


# --- FUNGSI SIMULASI CRAWLING ---
def simulate_x_crawling(limit):
    """
    Fungsi simulasi untuk membuat DataFrame hasil crawling tanpa keyword,
    mewakili akses 'full content' yang terbatas oleh limit.
    """
    st.warning("⚠️ **Mode Simulasi Aktif:** Aplikasi ini tidak melakukan crawling X/Twitter secara real-time. Data di bawah adalah data placeholder, mewakili akses 'full content' yang diizinkan token Anda.")
    
    if limit == 0:
        return pd.DataFrame()

    # Data dummy tweets yang bervariasi (placeholder untuk full content)
    base_tweets = [
        "Natural Language Processing (NLP) adalah bidang kecerdasan buatan yang menarik.",
        "Update terkini mengenai pasar saham global hari ini, terlihat adanya peningkatan signifikan.",
        "Selamat pagi dunia! Jangan lupa sarapan dan semangat untuk hari yang baru.",
        "Review film terbaru: alur cerita yang dalam, sinematografi yang memukau. Sangat direkomendasikan!",
        "Hari ini kita akan belajar tentang implementasi TF-IDF dan Bag of Word dalam analisis teks.",
        "Teknik Informatika Universitas Pelita Bangsa membuka pendaftaran untuk semester genap.",
        "Kopi dingin dan laptop tua, kombinasi sempurna untuk coding di sore hari.",
        "Mengapa penting untuk selalu memverifikasi sumber informasi sebelum membagikannya di media sosial.",
        "Perkembangan teknologi blockchain terus berlanjut, dengan inovasi di berbagai sektor industri."
    ]

    data = {
        'conversation_id_str': [1741185994123429000 + i for i in range(limit)],
        'created_at': [datetime.now() - timedelta(hours=i*3) for i in range(limit)],
        'favorite_count': np.random.randint(0, 1000, size=limit),
        'full_text': [base_tweets[i % len(base_tweets)] for i in range(limit)],
        'username': [f'user_{np.random.randint(100, 999)}' for i in range(limit)],
        'retweet_count': np.random.randint(0, 500, size=limit)
    }

    df = pd.DataFrame(data)
    # Format created_at agar mirip output Colab
    df['created_at'] = df['created_at'].dt.strftime('%a %b %d %H:%M:%S +0000 %Y')
    
    return df

# --- FUNGSI DOWNLOAD ---
@st.cache_data
def convert_df_to_csv(df):
    """Konversi DataFrame ke CSV untuk didownload."""
    return df.to_csv(index=False, sep=',').encode('utf-8')


# --- FUNGSI PREPROCESSING ---
def tokenize_manual(text):
    """Tokenisasi manual menggunakan regex"""
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    return tokens

def preprocess_text(text):
    """Fungsi preprocessing teks"""
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


# ... (Fungsi run_analysis, generate_dummy_data, deep_learning_classification, 
#      pos_tagging_analysis, named_entity_recognition, read_uploaded_file 
#      DITEMPATKAN DI SINI) ...
# Karena terlalu panjang untuk ditampilkan, saya hanya akan menunjukkan fungsi main_app yang dimodifikasi.

# --- FUNGSI UTAMA APLIKASI STREAMLIT (Dimodifikasi) ---
def main_app():
    """Fungsi utama untuk menjalankan aplikasi Streamlit"""
    st.set_page_config(
        page_title="NLP Toolkit: Representasi & Deep Learning",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 NLP TOOLKIT: Representasi Teks & Deep Learning")
    st.markdown("Aplikasi demonstrasi untuk preprocessing, representasi teks (BoW, TF-IDF, Word2Vec), klasifikasi Deep Learning (RNN/LSTM), dan analisis lanjutan (POS Tagging, NER).")

    # --- SIDEBAR INPUT ---
    with st.sidebar:
        st.header("⚙️ Konfigurasi Input")
        
        input_type = st.radio(
            "Pilih Tipe Input:",
            ("Input Manual", "Upload File", "X/Twitter Full Content (Simulasi)", "Web Scraping"),
            index=2 
        )
        
        list_dokumen_mentah = st.session_state.get('list_dokumen_mentah', [])
        
        if input_type != st.session_state.get('last_input_type', 'Input Manual'):
             list_dokumen_mentah = []
             st.session_state.list_dokumen_mentah = []
             st.session_state.list_dokumen_bersih = [] 
        
        st.session_state.last_input_type = input_type


        if input_type == "Input Manual":
            st.subheader("📝 Input Teks Manual")
            manual_text = st.text_area(
                "Masukkan satu atau lebih dokumen (pisahkan dengan baris baru ganda, gunakan bahasa Inggris untuk hasil NER/POS optimal):",
                "Jakarta is the capital city of Indonesia. I love this city.\n\n"
                "Natural Language Processing (NLP) is a field of artificial intelligence.",
                key="manual_input_text"
            )
            if manual_text:
                list_dokumen_mentah = [doc.strip() for doc in manual_text.split('\n\n') if doc.strip()]
        
        elif input_type == "Upload File":
            st.subheader("📁 Upload Dokumen")
            st.info("Gunakan opsi ini untuk upload `data_tweet.csv` hasil crawling Colab.")
            uploaded_file = st.file_uploader(
                "Upload file (.txt, .docx, .pdf, .csv)",
                type=["txt", "docx", "pdf", "csv"],
                accept_multiple_files=True,
                key="file_uploader_general"
            )
            
            if uploaded_file:
                list_dokumen_mentah = []
                for file in uploaded_file:
                    if file.name.endswith('.csv'):
                         try:
                            df_csv = pd.read_csv(file)
                            st.success(f"✅ Berhasil memuat file CSV: {file.name}")
                            if 'full_text' in df_csv.columns:
                                list_dokumen_mentah.extend(df_csv['full_text'].astype(str).dropna().tolist())
                            else:
                                st.error(f"❌ Kolom 'full_text' tidak ditemukan di {file.name}")
                         except Exception as e:
                             st.error(f"❌ Gagal membaca CSV {file.name}: {e}")
                    else:
                        content = read_uploaded_file(file)
                        if content:
                            list_dokumen_mentah.append(content)
                if list_dokumen_mentah:
                    st.success(f"✅ Total {len(list_dokumen_mentah)} dokumen/tweet berhasil dimuat.")

        elif input_type == "X/Twitter Full Content (Simulasi)":
            st.subheader("🔗 X/Twitter Crawling (Akses Token)")
            st.markdown("Masukkan token untuk mensimulasikan akses ke konten X/Twitter secara luas (tanpa keyword).")
            
            # --- INPUT TOKEN & PARAMETER CRAWLING ---
            auth_token_input = st.text_input(
                "Masukkan X/Twitter Auth Token Anda:",
                type="password",
                key="auth_token_input"
            )
            
            # Hapus input keyword
            limit = st.slider("Jumlah Konten yang Diambil (Limit)", 10, 500, 50)
            
            if st.button("🚀 Mulai Akses & Analisis (Simulasi)"):
                if not auth_token_input:
                    st.error("❌ Auth Token diperlukan untuk memulai simulasi akses konten.")
                else:
                    # --- EKSEKUSI SIMULASI CRAWLING ---
                    st.info(f"🔄 Mensimulasikan akses ke konten penuh X/Twitter dengan Limit: {limit} data.")
                    
                    # Panggil fungsi simulasi tanpa keyword
                    df_hasil_crawl = simulate_x_crawling(limit)
                    
                    if not df_hasil_crawl.empty:
                        list_dokumen_mentah = df_hasil_crawl['full_text'].astype(str).dropna().tolist()
                        st.session_state.list_dokumen_mentah = list_dokumen_mentah
                        st.session_state.df_hasil_crawl = df_hasil_crawl
                        st.session_state.list_dokumen_bersih = []
                        
                        st.success(f"✅ Simulasi Akses Konten Selesai! Ditemukan {len(list_dokumen_mentah)} tweet.")
                        st.dataframe(df_hasil_crawl.head(5))

                        # --- TOMBOL DOWNLOAD HASIL ---
                        csv = convert_df_to_csv(df_hasil_crawl)
                        st.download_button(
                            label="⬇️ Download Hasil Crawling (CSV)",
                            data=csv,
                            file_name=f'data_tweet_full_content_{datetime.now().strftime("%Y%m%d")}.csv',
                            mime='text/csv',
                            type="primary"
                        )
                        # --- AKHIR TOMBOL DOWNLOAD ---
                        
                    else:
                        st.warning("⚠️ Simulasi tidak menghasilkan data.")
            
        elif input_type == "Web Scraping":
            st.subheader("🔗 Web Scraping")
            url = st.text_input("Masukkan URL (Contoh: https://en.wikipedia.org/wiki/NLP):", 
                                "https://en.wikipedia.org/wiki/Natural_language_processing",
                                key="url_input")
            
            if st.button("Scrape Konten"):
                if url:
                    try:
                        st.info(f"🔄 Mengambil konten dari: **{url}**")
                        response = requests.get(url, timeout=10)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        paragraphs = soup.find_all('p')
                        scraped_text = '\n'.join([p.get_text() for p in paragraphs])
                        
                        if not scraped_text.strip():
                            st.warning("⚠️ Tidak ada konten paragraf yang ditemukan.")
                            list_dokumen_mentah = []
                        else:
                            list_dokumen_mentah = [scraped_text]
                            st.session_state.list_dokumen_mentah = list_dokumen_mentah
                            st.session_state.list_dokumen_bersih = []
                            st.success("✅ Web Scraping selesai.")
                            st.text_area("Konten yang di-Scrape (Snippet):", scraped_text[:1000] + "...", height=200)

                    except requests.exceptions.Timeout:
                        st.error("❌ Timeout: Permintaan melebihi batas waktu.")
                        list_dokumen_mentah = []
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Gagal koneksi ke URL: {e}")
                        list_dokumen_mentah = []
                    except Exception as e:
                        st.error(f"❌ Gagal melakukan web scraping: {e}")
                        list_dokumen_mentah = []

        
        if input_type == "Input Manual" or input_type == "Upload File":
             st.session_state.list_dokumen_mentah = list_dokumen_mentah

        st.markdown("---")
        st.subheader("🗂️ Status Dokumen")
        st.metric("Jumlah Dokumen Mentah", len(st.session_state.list_dokumen_mentah))
        
        if len(st.session_state.list_dokumen_mentah) == 0:
            st.warning("Mohon masukkan dokumen untuk memulai analisis.")


    # --- TABS UTAMA ---
    tab_preprocess, tab_repr, tab_dl, tab_advanced = st.tabs([
        "1. Preprocessing", 
        "2. Representasi Teks", 
        "3. Deep Learning (Klasifikasi)", 
        "4. Analisis Lanjutan (POS/NER)"
    ])
    
    # ... (Kode untuk setiap tab tetap sama seperti di jawaban sebelumnya) ...

# --- RUN APP ---
if __name__ == '__main__':
    # Inisialisasi session state
    if 'list_dokumen_bersih' not in st.session_state:
        st.session_state.list_dokumen_bersih = []
    if 'list_dokumen_mentah' not in st.session_state:
        st.session_state.list_dokumen_mentah = []
    if 'last_input_type' not in st.session_state:
        st.session_state.last_input_type = 'Input Manual'
    
    # Run main application
    main_app()
