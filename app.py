# =================================================================
# NLP TOOLKIT DENGAN INTEGRASI TWITTER CRAWLER
# Aplikasi Streamlit untuk Analisis Teks & Crawling Twitter Real-time
# =================================================================

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
import subprocess
import json
from pathlib import Path

# Preprocessing
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

# Representasi Teks
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Document Processing
import docx
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

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
    nltk_packages = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    
    for package in nltk_packages:
        for path in ['tokenizers', 'taggers', 'chunkers', 'corpora']:
            try:
                nltk.data.find(f'{path}/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass

try:
    download_nltk_data()
except Exception as e:
    st.warning(f"⚠️ Beberapa data NLTK gagal didownload: {e}")


# --- INISIALISASI MODEL SASTRAWI ---
@st.cache_resource
def inisialisasi_model():
    """Menginisialisasi model Sastrawi (Stemmer & Stopword Remover)"""
    if not SASTRAWI_AVAILABLE:
        return None, None
    try:
        stemmer_factory = StemmerFactory()
        stemmer = stemmer_factory.create_stemmer()
        
        stopword_factory = StopWordRemoverFactory()
        stopword_remover = stopword_factory.create_stop_word_remover()
        return stemmer, stopword_remover
    except Exception as e:
        st.error(f"❌ Gagal memuat model Sastrawi: {e}")
        return None, None

stemmer, stopword_remover = inisialisasi_model()


# --- FUNGSI TWITTER CRAWLING REAL ---
def check_node_installation():
    """Cek apakah Node.js sudah terinstall"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def install_nodejs():
    """Instalasi Node.js untuk sistem Linux (Google Colab compatible)"""
    try:
        st.info("🔧 Mencoba menginstal Node.js...")
        
        # Cek apakah punya sudo access
        has_sudo = subprocess.run(['which', 'sudo'], capture_output=True).returncode == 0
        
        if not has_sudo:
            st.warning("⚠️ Tidak ada sudo access. Node.js harus diinstall secara manual atau gunakan mode simulasi.")
            return False
        
        # Install Node.js dengan sudo
        subprocess.run(['sudo', 'curl', '-fsSL', 'https://deb.nodesource.com/setup_18.x', 
                       '-o', '/tmp/nodesource_setup.sh'], check=True, capture_output=True, timeout=30)
        subprocess.run(['sudo', 'bash', '/tmp/nodesource_setup.sh'], check=True, capture_output=True, timeout=60)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'nodejs'], check=True, capture_output=True, timeout=120)
        
        st.success("✅ Node.js berhasil diinstal!")
        return True
        
    except subprocess.TimeoutExpired:
        st.warning("⚠️ Instalasi Node.js timeout. Gunakan mode simulasi.")
        return False
    except Exception as e:
        st.warning(f"⚠️ Instalasi Node.js gagal: {str(e)[:100]}")
        return False


def run_tweet_harvest(auth_token, search_query, limit, output_file):
    """Jalankan tweet-harvest untuk crawling Twitter"""
    try:
        # Buat folder output jika belum ada
        output_dir = Path("tweets-data")
        output_dir.mkdir(exist_ok=True)
        
        # Build command
        cmd = [
            'npx', '--yes', 'tweet-harvest',
            '-o', output_file,
            '-l', str(limit),
            '--token', auth_token
        ]
        
        if search_query:
            cmd.extend(['-s', search_query])
        
        # Jalankan command
        st.info("⏳ Memulai crawling Twitter...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 menit timeout
        )
        
        if result.returncode == 0:
            return True, "Crawling berhasil!"
        else:
            return False, f"Error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout: Proses crawling melebihi batas waktu (5 menit)"
    except Exception as e:
        return False, f"Error: {str(e)}"


def real_twitter_crawling(auth_token, search_query, limit):
    """Fungsi utama untuk crawling Twitter real"""
    
    # Cek Node.js
    if not check_node_installation():
        st.warning("⚠️ Node.js belum terinstall.")
        
        # Tampilkan info platform
        import platform
        os_type = platform.system()
        
        if os_type == "Linux":
            st.info("📦 Platform: Linux - Mencoba instalasi otomatis...")
            if not install_nodejs():
                st.error("❌ Instalasi gagal. Silakan gunakan **Mode Simulasi** atau install Node.js secara manual.")
                st.info("💡 **Cara install manual:**\n```bash\nsudo apt-get update\nsudo apt-get install -y nodejs npm\n```")
                return None
        else:
            st.warning(f"📦 Platform: {os_type}")
            st.info("💡 **Untuk crawling real, install Node.js terlebih dahulu:**")
            st.markdown("- **Windows/Mac**: Download dari [nodejs.org](https://nodejs.org/)")
            st.markdown("- Setelah install, restart aplikasi ini")
            st.markdown("- Atau gunakan **Mode Simulasi** untuk testing")
            return None
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"twitter_data_{timestamp}.csv"
    output_path = f"tweets-data/{output_file}"
    
    # Jalankan crawling
    success, message = run_tweet_harvest(auth_token, search_query, limit, output_file)
    
    if success:
        # Baca hasil
        if os.path.exists(output_path):
            try:
                df = pd.read_csv(output_path)
                
                if df.empty:
                    st.error("❌ File CSV kosong. Kemungkinan token tidak valid atau tidak ada data.")
                    st.info("💡 **Tips:**\n- Pastikan token masih fresh\n- Coba dengan limit lebih kecil (10-20)\n- Gunakan keyword yang lebih spesifik")
                    return None
                
                st.success(f"✅ Berhasil crawling {len(df)} tweets!")
                return df
                
            except Exception as e:
                st.error(f"❌ Error membaca file CSV: {e}")
                return None
        else:
            st.error(f"❌ File output tidak ditemukan: {output_path}")
            return None
    else:
        st.error(f"❌ Crawling gagal: {message}")
        st.info("💡 Kemungkinan penyebab:\n- Token tidak valid/expired\n- Rate limit Twitter\n- Koneksi internet bermasalah")
        return None


# --- FUNGSI SIMULASI CRAWLING (FALLBACK) ---
def simulate_x_crawling(limit):
    """Fungsi simulasi untuk fallback mode"""
    st.warning("⚠️ **Mode Simulasi:** Data di bawah adalah placeholder.")
    
    if limit == 0:
        return pd.DataFrame()

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
    if not SASTRAWI_AVAILABLE or stemmer is None:
        st.error("❌ Preprocessing dinonaktifkan karena library Sastrawi tidak terinstal.")
        return text.lower()
        
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


# --- FUNGSI ANALISIS REPRESENTASI TEKS ---
def run_analysis(list_dokumen_bersih):
    """Menganalisis dokumen dengan BoW, TF-IDF, dan Word2Vec"""
    st.header("📊 IMPLEMENTASI REPRESENTASI TEKS")
    st.markdown("---")

    if not list_dokumen_bersih or not all(doc.strip() for doc in list_dokumen_bersih):
        st.error("❌ Dokumen bersih kosong. Cek hasil preprocessing.")
        return

    # BAG OF WORDS
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
        st.info("💡 **Penjelasan:** Menghitung frekuensi kemunculan setiap kata unik dalam dokumen.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Kata Unik", len(fitur_bow))
        with col2:
            st.metric("Total Kemunculan Kata", int(bow_matrix.sum()))
            
    except ValueError as e:
        st.error(f"❌ ERROR BoW: {e}")

    st.markdown("---")

    # TF-IDF
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
        st.info("💡 **Penjelasan:** Memberi bobot kata berdasarkan frekuensi dan kelangkaannya.")
        
        if len(fitur_tfidf) > 0:
            max_tfidf = df_tfidf.max().sort_values(ascending=False).head(5)
            st.write("**Top 5 Kata Berdasarkan Bobot TF-IDF:**")
            st.bar_chart(max_tfidf)
        
    except ValueError as e:
        st.error(f"❌ ERROR TF-IDF: {e}")

    st.markdown("---")

    # WORD2VEC
    st.subheader("3️⃣ Metode Word2Vec")
    try:
        tokenized_docs_w2v = [doc.split() for doc in list_dokumen_bersih if doc.strip()]
        
        if not tokenized_docs_w2v:
            st.error("❌ Tidak ada token untuk dilatih.")
            return

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
        
        vocab_size = len(model_w2v.wv.index_to_key)
        st.metric("Ukuran Vocabulary", vocab_size)
        
        if model_w2v.wv.index_to_key:
            kata_uji = model_w2v.wv.index_to_key[0] 
            
            st.write(f"**Contoh Vektor untuk kata '{kata_uji}'** (10 dimensi pertama):")
            vektor = model_w2v.wv[kata_uji][:10].tolist()
            st.json({f"dim_{i+1}": round(v, 4) for i, v in enumerate(vektor)})

            try:
                kata_mirip = model_w2v.wv.most_similar(kata_uji, topn=5)
                st.write(f"**Kata mirip dengan '{kata_uji}':**")
                
                df_mirip = pd.DataFrame(kata_mirip, columns=['Kata', 'Similarity Score'])
                st.dataframe(df_mirip, use_container_width=True)
                
            except (KeyError, IndexError):
                st.warning(f"⚠️ Tidak ada kata mirip untuk '{kata_uji}'")
        else:
            st.error("❌ Vocabulary Word2Vec kosong.")
            
    except Exception as e:
        st.error(f"❌ ERROR Word2Vec: {e}")


# --- FUNGSI DEEP LEARNING ---
def generate_dummy_data():
    """Generate data dummy untuk demonstrasi"""
    positive_texts = [
        "this movie is amazing and fantastic", "i love this product it works great",
        "excellent service highly recommend", "best experience ever very satisfied",
        "wonderful performance absolutely brilliant", "incredible quality worth every penny",
        "outstanding results exceeded expectations", "fantastic features easy to use",
        "great value for money", "superb craftsmanship top notch",
        "impressive design beautiful look", "remarkable improvement love it",
        "perfect solution exactly what needed", "exceptional quality highly pleased",
        "amazing results very happy", "this is truly a masterpiece",
        "so much fun to watch it again", "absolutely delightful time",
        "the finest quality you can buy", "highly satisfied with my purchase"
    ]
    
    negative_texts = [
        "this is terrible very disappointed", "worst product ever waste of money",
        "horrible experience never again", "poor quality not recommended",
        "awful service very unhappy", "completely useless total disaster",
        "disappointing results not worth it", "terrible design badly made",
        "worst purchase regret buying", "very poor performance failed",
        "bad quality broke quickly", "unsatisfactory service very poor",
        "horrible experience waste time", "disappointing product not good",
        "terrible service very bad", "i truly hate this product",
        "never buy this item again", "waste of time and energy",
        "not worth the attention it gets", "i feel cheated by this company"
    ]
    
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    return texts, np.array(labels)


def deep_learning_classification(texts, labels):
    """Klasifikasi teks menggunakan RNN/LSTM"""
    if not DEEP_LEARNING_AVAILABLE:
        st.error("❌ Library Deep Learning tidak tersedia.")
        return
        
    st.header("🤖 DEEP LEARNING - RNN/LSTM CLASSIFICATION")
    st.info("Model memetakan kata menjadi vektor dan mempelajari urutan kata.")
    
    if len(texts) < 10:
        st.warning("⚠️ Data terlalu sedikit untuk training.")
        return
    
    # PREPROCESSING
    st.subheader("1️⃣ Preprocessing Data")
    
    with st.spinner("Memproses teks..."):
        max_words = 5000 
        max_len = 100     
        
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        vocab_size = len(tokenizer.word_index) + 1
        st.success(f"✅ Preprocessing selesai! Vocabulary: {vocab_size}")
    
    # SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42
    )
    
    st.subheader("2️⃣ Desain Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Pilih Arsitektur:",
            ["Simple RNN", "LSTM", "Bidirectional LSTM"]
        )
    
    with col2:
        epochs = st.slider("Jumlah Epochs:", 5, 50, 10)
    
    # BUILD MODEL
    with st.spinner(f"Membangun model {model_type}..."):
        model = Sequential()
        embedding_dim = 128
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        
        if model_type == "Simple RNN":
            model.add(SimpleRNN(64, return_sequences=False))
        elif model_type == "LSTM":
            model.add(LSTM(64, return_sequences=False))
        elif model_type == "Bidirectional LSTM":
            model.add(Bidirectional(LSTM(64, return_sequences=False))) 
        
        model.add(Dropout(0.5)) 
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    st.success(f"✅ Model {model_type} siap!")
    
    if st.button("🚀 Mulai Training", type="primary"):
        with st.spinner(f"Training {epochs} epochs..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f} - Acc: {logs['accuracy']:.4f}")
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[StreamlitCallback()]
            )
            
            progress_bar.empty()
            status_text.empty()
        
        st.success("✅ Training selesai!")
        
        # Plot History
        st.subheader("📈 Training History")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()
        st.pyplot(fig)
        
        # EVALUASI
        st.subheader("📊 Evaluasi Model")
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("📊 F1-Score", f"{f1:.4f}")
        
        target_names = ['Negative', 'Positive']
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4))
        
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=target_names, yticklabels=target_names)
        st.pyplot(fig)
        
        st.session_state.trained_model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.max_len = max_len


# --- FUNGSI ANALISIS LANJUTAN ---
def pos_tagging_analysis(list_dokumen_mentah):
    """POS Tagging Analysis"""
    st.header("🏷️ POS TAGGING")
    st.info("💡 Memberikan label kategori gramatikal pada setiap kata")
    
    for idx, doc in enumerate(list_dokumen_mentah[:3]):  # Limit 3 dokumen
        with st.expander(f"📄 Dokumen {idx+1}", expanded=(idx==0)):
            try:
                tokens = word_tokenize(doc.lower())
                if not tokens: 
                    continue
                pos_tags = pos_tag(tokens)
                df_pos = pd.DataFrame(pos_tags, columns=['Word', 'POS Tag'])
                st.dataframe(df_pos, use_container_width=True, height=300)
                
                pos_counts = df_pos['POS Tag'].value_counts()
                st.bar_chart(pos_counts.head(10))
                
            except Exception as e:
                st.error(f"❌ Error: {e}")


def named_entity_recognition(list_dokumen_mentah):
    """Named Entity Recognition"""
    st.header("🎯 NAMED ENTITY RECOGNITION")
    st.info("💡 Identifikasi entitas bernama (orang, lokasi, organisasi)")
    
    for idx, doc in enumerate(list_dokumen_mentah[:3]):  # Limit 3 dokumen
        with st.expander(f"📄 Dokumen {idx+1}", expanded=(idx==0)):
            try:
                tokens = word_tokenize(doc)
                pos_tags = pos_tag(tokens)
                named_entities = ne_chunk(pos_tags, binary=False)
                
                entities_dict = {'PERSON': [], 'LOCATION': [], 'ORGANIZATION': [], 'GPE': []}
                
                for chunk in named_entities:
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join(c[0] for c in chunk)
                        entity_label = chunk.label()
                        if entity_label in entities_dict:
                            entities_dict[entity_label].append(entity_text)
                
                found = False
                for category, entities in entities_dict.items():
                    if entities:
                        found = True
                        st.subheader(category)
                        st.write("- " + "\n- ".join(sorted(set(entities))))
                
                if not found:
                    st.warning("⚠️ Tidak ada entitas ditemukan")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")


# --- FUNGSI BACA FILE ---
@st.cache_data
def read_uploaded_file(uploaded_file):
    """Membaca file upload"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    text_content = ""

    try:
        if file_extension == 'txt':
            text_content = uploaded_file.getvalue().decode("utf-8")
        elif file_extension == 'docx':
            doc = docx.Document(uploaded_file)
            paragraphs = [p.text for p in doc.paragraphs]
            text_content = '\n'.join(paragraphs)
        elif file_extension == 'pdf':
            if not PDF_AVAILABLE:
                st.warning("⚠️ Library pdfplumber tidak tersedia.")
                return None
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        text_content += page.extract_text() if page.extract_text() else ""
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path) 
        else:
            return None
            
        return text_content
        
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        return None


# --- APLIKASI UTAMA ---
def main_app():
    """Fungsi utama aplikasi"""
    st.set_page_config(
        page_title="NLP Toolkit + Twitter Crawler",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 NLP TOOLKIT + TWITTER CRAWLER")
    st.markdown("Analisis teks dengan integrasi crawling Twitter real-time")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Konfigurasi Input")
        
        input_type = st.radio(
            "Pilih Tipe Input:",
            ("Twitter Crawling (Real)", "Input Manual", "Upload File", "Web Scraping"),
            index=0
        )
        
        list_dokumen_mentah = st.session_state.get('list_dokumen_mentah', [])
        
        if input_type != st.session_state.get('last_input_type', 'Twitter Crawling (Real)'):
             list_dokumen_mentah = []
             st.session_state.list_dokumen_mentah = []
             st.session_state.list_dokumen_bersih = []
        
        st.session_state.last_input_type = input_type

        # --- TWITTER CRAWLING REAL ---
        if input_type == "Twitter Crawling (Real)":
            st.subheader("🐦 Twitter Crawler")
            
            # Panduan Token
            with st.expander("📖 Cara Mendapatkan Auth Token", expanded=False):
                st.markdown("""
                **LANGKAH-LANGKAH:**
                1. Buka Twitter/X di browser
                2. Login ke akun Anda
                3. Tekan F12 atau klik kanan → Inspect
                4. Buka tab **Application** (Chrome) / **Storage** (Firefox)
                5. Expand **Cookies** → Pilih https://twitter.com
                6. Cari cookie **auth_token**
                7. Copy VALUE-nya (bukan nama!)
                8. Paste di input di bawah
                
                ⚠️ **JANGAN bagikan token ke siapa pun!**
                """)
            
            auth_token = st.text_input(
                "Auth Token:",
                type="password",
                help="Token dari cookies Twitter Anda"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                crawl_mode = st.selectbox(
                    "Mode Crawling:",
                    ["Timeline (Tanpa Filter)", "Username Spesifik", "Keyword Search"]
                )
            
            with col2:
                limit = st.number_input("Jumlah Tweet:", 10, 1000, 100)
            
            search_query = ""
            if crawl_mode == "Username Spesifik":
                username = st.text_input("Username (tanpa @):", "elonmusk")
                search_query = f"from:{username}"
            elif crawl_mode == "Keyword Search":
                keyword = st.text_input("Keyword:", "python")
                search_query = keyword
            
            use_simulation = st.checkbox(
                "🎭 Gunakan Mode Simulasi (Tanpa Node.js)", 
                value=True,
                help="Aktifkan ini jika Node.js tidak terinstall atau untuk testing cepat"
            )
            
            if st.button("🚀 Mulai Crawling & Analisis", type="primary"):
                if not auth_token and not use_simulation:
                    st.error("❌ Token diperlukan untuk crawling real!")
                else:
                    with st.spinner("⏳ Memproses..."):
                        if use_simulation:
                            df_hasil = simulate_x_crawling(limit)
                        else:
                            df_hasil = real_twitter_crawling(auth_token, search_query, limit)
                        
                        if df_hasil is not None and not df_hasil.empty:
                            # Extract teks
                            if 'full_text' in df_hasil.columns:
                                list_dokumen_mentah = df_hasil['full_text'].astype(str).dropna().tolist()
                            elif 'text' in df_hasil.columns:
                                list_dokumen_mentah = df_hasil['text'].astype(str).dropna().tolist()
                            else:
                                st.error("❌ Kolom teks tidak ditemukan")
                                list_dokumen_mentah = []
                            
                            st.session_state.list_dokumen_mentah = list_dokumen_mentah
                            st.session_state.df_hasil_crawl = df_hasil
                            st.session_state.list_dokumen_bersih = []
                            
                            st.success(f"✅ Berhasil! {len(list_dokumen_mentah)} tweets dimuat")
                            st.dataframe(df_hasil.head(5))
                            
                            # Download button
                            csv = convert_df_to_csv(df_hasil)
                            st.download_button(
                                "⬇️ Download CSV",
                                data=csv,
                                file_name=f'twitter_{datetime.now().strftime("%Y%m%d")}.csv',
                                mime='text/csv'
                            )

        # --- INPUT MANUAL ---
        elif input_type == "Input Manual":
            st.subheader("📝 Input Manual")
            manual_text = st.text_area(
                "Masukkan teks (pisahkan dengan baris baru ganda):",
                "Jakarta is the capital city of Indonesia.\n\n"
                "Natural Language Processing is fascinating.",
                height=200
            )
            if manual_text:
                list_dokumen_mentah = [doc.strip() for doc in manual_text.split('\n\n') if doc.strip()]
                st.session_state.list_dokumen_mentah = list_dokumen_mentah
        
        # --- UPLOAD FILE ---
        elif input_type == "Upload File":
            st.subheader("📁 Upload File")
            uploaded_files = st.file_uploader(
                "Upload file (.csv, .txt, .docx, .pdf):",
                type=["txt", "docx", "pdf", "csv"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                list_dokumen_mentah = []
                for file in uploaded_files:
                    if file.name.endswith('.csv'):
                        try:
                            df_csv = pd.read_csv(file)
                            if 'full_text' in df_csv.columns:
                                list_dokumen_mentah.extend(df_csv['full_text'].astype(str).dropna().tolist())
                            elif 'text' in df_csv.columns:
                                list_dokumen_mentah.extend(df_csv['text'].astype(str).dropna().tolist())
                            else:
                                st.error(f"❌ Kolom teks tidak ditemukan di {file.name}")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        content = read_uploaded_file(file)
                        if content:
                            list_dokumen_mentah.append(content)
                
                st.session_state.list_dokumen_mentah = list_dokumen_mentah
                if list_dokumen_mentah:
                    st.success(f"✅ {len(list_dokumen_mentah)} dokumen dimuat")

        # --- WEB SCRAPING ---
        elif input_type == "Web Scraping":
            st.subheader("🔗 Web Scraping")
            url = st.text_input("URL:", "https://en.wikipedia.org/wiki/Natural_language_processing")
            
            if st.button("Scrape"):
                if url:
                    try:
                        with st.spinner("⏳ Scraping..."):
                            response = requests.get(url, timeout=10)
                            soup = BeautifulSoup(response.text, 'html.parser')
                            paragraphs = soup.find_all('p')
                            scraped_text = '\n'.join([p.get_text() for p in paragraphs])
                            
                            if scraped_text.strip():
                                list_dokumen_mentah = [scraped_text]
                                st.session_state.list_dokumen_mentah = list_dokumen_mentah
                                st.session_state.list_dokumen_bersih = []
                                st.success("✅ Scraping selesai")
                                st.text_area("Preview:", scraped_text[:500] + "...", height=150)
                            else:
                                st.warning("⚠️ Tidak ada konten ditemukan")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        st.markdown("---")
        st.metric("📊 Dokumen Dimuat", len(st.session_state.list_dokumen_mentah))

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Preprocessing", 
        "2️⃣ Representasi Teks", 
        "3️⃣ Deep Learning", 
        "4️⃣ Analisis Lanjutan"
    ])

    # TAB 1: PREPROCESSING
    with tab1:
        st.header("PREPROCESSING")
        list_docs = st.session_state.get('list_dokumen_mentah', [])
        
        if list_docs:
            if st.button("✨ Mulai Preprocessing", type="primary"):
                list_clean = []
                with st.spinner("Memproses..."):
                    for idx, doc in enumerate(list_docs[:10]):  # Limit 10
                        st.subheader(f"Dokumen {idx+1}")
                        with st.expander("Teks Mentah"):
                            st.text(doc[:500])
                        
                        cleaned = preprocess_text(doc)
                        if cleaned:
                            list_clean.append(cleaned)
                            st.code(cleaned[:300], language="text")
                
                st.session_state.list_dokumen_bersih = list_clean
                st.success(f"✅ {len(list_clean)} dokumen diproses!")
        else:
            st.warning("Input dokumen terlebih dahulu")

    # TAB 2: REPRESENTASI TEKS
    with tab2:
        list_clean = st.session_state.get('list_dokumen_bersih', [])
        if list_clean:
            run_analysis(list_clean)
        else:
            st.warning("Lakukan preprocessing terlebih dahulu")

    # TAB 3: DEEP LEARNING
    with tab3:
        if DEEP_LEARNING_AVAILABLE:
            st.info("💡 Menggunakan data dummy untuk demonstrasi")
            
            if st.button("▶️ Jalankan Demo", type="primary"):
                texts, labels = generate_dummy_data()
                deep_learning_classification(texts, labels)
            
            st.markdown("---")
            st.subheader("🔮 Prediksi Teks Baru")
            
            if 'trained_model' in st.session_state:
                new_text = st.text_area("Masukkan teks:", "This product is terrible")
                
                if st.button("Prediksi"):
                    if new_text.strip():
                        new_seq = st.session_state.tokenizer.texts_to_sequences([new_text])
                        new_padded = pad_sequences(new_seq, maxlen=st.session_state.max_len, 
                                                   padding='post', truncating='post')
                        
                        pred = st.session_state.trained_model.predict(new_padded, verbose=0)
                        label = "Positive" if pred[0] > 0.5 else "Negative"
                        conf = float(pred[0]) if pred[0] > 0.5 else 1 - float(pred[0])
                        
                        st.success(f"**Prediksi:** {label}")
                        st.info(f"**Confidence:** {conf:.2%}")
            else:
                st.warning("Jalankan demo terlebih dahulu")
        else:
            st.error("❌ TensorFlow tidak tersedia")

    # TAB 4: ANALISIS LANJUTAN
    with tab4:
        list_docs = st.session_state.get('list_dokumen_mentah', [])
        
        if list_docs:
            st.info("💡 Analisis ini lebih efektif untuk teks Bahasa Inggris")
            
            st.markdown("---")
            pos_tagging_analysis(list_docs)
            
            st.markdown("---")
            named_entity_recognition(list_docs)
        else:
            st.warning("Input dokumen terlebih dahulu")


# --- JALANKAN APLIKASI ---
if __name__ == '__main__':
    # Inisialisasi session state
    if 'list_dokumen_bersih' not in st.session_state:
        st.session_state.list_dokumen_bersih = []
    if 'list_dokumen_mentah' not in st.session_state:
        st.session_state.list_dokumen_mentah = []
    if 'last_input_type' not in st.session_state:
        st.session_state.last_input_type = 'Twitter Crawling (Real)'
    
    main_app()
