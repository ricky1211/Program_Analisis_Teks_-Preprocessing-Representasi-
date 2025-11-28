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
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
except ImportError:
    st.warning("⚠️ Library Sastrawi tidak ditemukan. Fungsi Stemming dan Stopword Removal akan dinonaktifkan.")

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
    try:
        stemmer_factory = StemmerFactory()
        stemmer = stemmer_factory.create_stemmer()
        
        stopword_factory = StopWordRemoverFactory()
        stopword_remover = stopword_factory.create_stop_word_remover()
        return stemmer, stopword_remover
    except NameError:
        # Jika Sastrawi tidak terimport
        return None, None

try:
    stemmer, stopword_remover = inisialisasi_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model Sastrawi: {e}")
    stemmer, stopword_remover = None, None


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
    if stemmer is None:
        st.error("❌ Preprocessing (Stemming/Stopword) dinonaktifkan karena library Sastrawi tidak terinstal.")
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
        st.error("❌ Dokumen bersih kosong atau hanya berisi spasi. Cek hasil preprocessing.")
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
        st.info("💡 **Penjelasan:** Menghitung frekuensi kemunculan setiap kata unik (*term frequency*) dalam dokumen.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Kata Unik (Vocabulary Size)", len(fitur_bow))
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
        st.info("💡 **Penjelasan:** Memberi bobot kata berdasarkan frekuensi (TF) dan kelangkaannya (IDF) di seluruh dokumen.")
        
        if len(fitur_tfidf) > 0:
            max_tfidf = df_tfidf.max().sort_values(ascending=False).head(5)
            st.write("**Top 5 Kata Berdasarkan Bobot TF-IDF Tertinggi:**")
            st.bar_chart(max_tfidf)
        
    except ValueError as e:
        st.error(f"❌ ERROR TF-IDF: {e}")

    st.markdown("---")

    # WORD2VEC
    st.subheader("3️⃣ Metode Word2Vec")
    try:
        tokenized_docs_w2v = [doc.split() for doc in list_dokumen_bersih if doc.strip()]
        
        if not tokenized_docs_w2v:
            st.error("❌ ERROR: Tidak ada token untuk dilatih.")
            return

        st.write("**Input untuk Word2Vec (Sample):**")
        sample_code = [doc[:50] for doc in tokenized_docs_w2v[:2]]
        st.code(str(sample_code).replace('\'', '"'), language="python")
        
        with st.spinner("🔄 Melatih model Word2Vec (Vector Size=100, Window=5)..."):
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
                st.write(f"**Kata yang paling mirip/berdekatan dengan '{kata_uji}' (Semantic Similarity):**")
                
                df_mirip = pd.DataFrame(kata_mirip, columns=['Kata', 'Similarity Score'])
                st.dataframe(df_mirip, use_container_width=True)
                
            except KeyError:
                st.warning(f"⚠️ Kata '{kata_uji}' tidak memiliki kata mirip.")
            except IndexError:
                st.warning("⚠️ Vocabulary Word2Vec terlalu kecil.")
        else:
            st.error("❌ Vocabulary Word2Vec kosong.")
            
    except Exception as e:
        st.error(f"❌ ERROR Word2Vec: {e}")


# --- FUNGSI DEEP LEARNING (RNN/LSTM) ---
def generate_dummy_data():
    """Generate data dummy untuk demonstrasi klasifikasi sentimen biner (0/1)"""
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
    """
    Klasifikasi teks menggunakan RNN/LSTM
    """
    if not DEEP_LEARNING_AVAILABLE:
        st.error("❌ Library Deep Learning (Tensorflow/Keras) tidak tersedia.")
        return
        
    st.header("🤖 DEEP LEARNING - RNN/LSTM CLASSIFICATION")
    
    st.info("Model memetakan kata menjadi vektor (Embedding) dan mempelajari urutan kata, cocok untuk klasifikasi.")
    
    if len(texts) < 10:
        st.warning("⚠️ Data terlalu sedikit untuk training.")
        return
    
    # PREPROCESSING UNTUK DEEP LEARNING
    st.subheader("1️⃣ Preprocessing Data")
    
    with st.spinner("Memproses teks..."):
        max_words = 5000 
        max_len = 100     
        
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        vocab_size = len(tokenizer.word_index) + 1
        st.success(f"✅ Preprocessing selesai! Vocabulary Size: {vocab_size}")
    
    # SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42
    )
    
    st.subheader("3️⃣ Desain Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Pilih Arsitektur Model:",
            ["Simple RNN", "LSTM", "Bidirectional LSTM"],
            key="model_type_select"
        )
    
    with col2:
        epochs = st.slider("Jumlah Epochs:", 5, 50, 10, key="epochs_slider")
    
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
        
        num_classes = len(np.unique(labels))
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss_fn = 'binary_crossentropy'
        else:
            model.add(Dense(num_classes, activation='softmax'))
            loss_fn = 'sparse_categorical_crossentropy'
        
        model.compile(
            optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy']
        )
    
    st.success(f"✅ Model {model_type} berhasil dibuat!")
    
    if st.button("🚀 Mulai Training", type="primary", key="start_training_button"):
        with st.spinner(f"Training model selama {epochs} epochs..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f} - Acc: {logs['accuracy']:.4f} - Val Loss: {logs['val_loss']:.4f} - Val Acc: {logs['val_accuracy']:.4f}")
            
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
        # ... (Kode plotting history) ...
        st.subheader("📈 Training History")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history.history['loss'], label='Training Loss'); ax1.plot(history.history['val_loss'], label='Validation Loss'); ax1.set_title('Model Loss'); ax1.legend()
        ax2.plot(history.history['accuracy'], label='Training Accuracy'); ax2.plot(history.history['val_accuracy'], label='Validation Accuracy'); ax2.set_title('Model Accuracy'); ax2.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        # EVALUASI MODEL
        st.subheader("5️⃣ Evaluasi Model")
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1) 
        
        st.metric("🎯 Accuracy", f"{accuracy:.4f}")
        st.metric("📊 F1-Score (Positive Class)", f"{f1:.4f}")
        
        target_names = ['Negative (0)', 'Positive (1)']
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4))
        
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=target_names, yticklabels=target_names)
        st.pyplot(fig)
        
        st.session_state.trained_model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.max_len = max_len
    
    st.markdown("---")


# --- FUNGSI ANALISIS LANJUTAN ---
def pos_tagging_analysis(list_dokumen_mentah):
    """A - POS Tagging: Analisis Part-of-Speech pada teks"""
    st.header("🏷️ A - POS TAGGING")
    st.info("💡 **Part-of-Speech Tagging** memberikan label kategori gramatikal pada setiap kata (kata benda, kata kerja, kata sifat, dll)")
    
    for idx, doc in enumerate(list_dokumen_mentah):
        with st.expander(f"📄 POS Tagging - Dokumen {idx+1}", expanded=(idx==0)):
            try:
                tokens = word_tokenize(doc.lower())
                if not tokens: continue
                pos_tags = pos_tag(tokens)
                df_pos = pd.DataFrame(pos_tags, columns=['Word', 'POS Tag'])
                st.dataframe(df_pos, use_container_width=True, height=300)
                
                pos_counts = df_pos['POS Tag'].value_counts()
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**Distribusi Top 10 POS Tags:**")
                    st.bar_chart(pos_counts.head(10))
                
                st.write("**📚 Penjelasan POS Tags yang Menarik (dari NLTK):**")
                pos_explanations = {
                    'NN': '**NN (Noun, singular)**', 'NNS': '**NNS (Noun, plural)**', 
                    'VB': '**VB (Verb, base form)**', 'VBD': '**VBD (Verb, past tense)**', 
                    'JJ': '**JJ (Adjective)**', 'RB': '**RB (Adverb)**', 
                    'PRP': '**PRP (Personal Pronoun)**', 'DT': '**DT (Determiner)**'
                }
                for pos in pos_counts.head(5).index:
                    if pos in pos_explanations:
                        st.markdown(f"- {pos_explanations[pos]}")
                
            except Exception as e:
                st.error(f"❌ Error POS Tagging: {e}")


def named_entity_recognition(list_dokumen_mentah):
    """B - Named Entity Recognition: Identifikasi entitas bernama"""
    st.header("🎯 B - NAMED ENTITY RECOGNITION (NER)")
    st.info("💡 **NER** mengidentifikasi dan mengklasifikasikan entitas bernama seperti nama orang, lokasi, organisasi, tanggal, dll. ")
    
    for idx, doc in enumerate(list_dokumen_mentah):
        with st.expander(f"📄 NER - Dokumen {idx+1}", expanded=(idx==0)):
            try:
                tokens = word_tokenize(doc)
                pos_tags = pos_tag(tokens)
                named_entities = ne_chunk(pos_tags, binary=False) 
                
                entities_dict = {'PERSON': [], 'LOCATION': [], 'ORGANIZATION': [], 'DATE': [], 'GPE': []}
                
                for chunk in named_entities:
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join(c[0] for c in chunk)
                        entity_label = chunk.label()
                        if entity_label in entities_dict:
                            entities_dict[entity_label].append(entity_text)
                
                st.write("**🔍 Entitas yang Ditemukan:**")
                found_entities = False
                for category, entities in entities_dict.items():
                    if entities:
                        found_entities = True
                        with st.container(border=True):
                            st.subheader(category)
                            unique_entities = sorted(list(set(entities)))
                            st.markdown("- " + "\n- ".join(unique_entities))
                
                if not found_entities:
                    st.warning("⚠️ Tidak ada entitas bernama yang ditemukan dalam dokumen ini.")
                
            except Exception as e:
                st.error(f"❌ Error NER: {e}")


# --- FUNGSI UTAMA UNTUK MEMBACA DOKUMEN (non-CSV) ---
@st.cache_data
def read_uploaded_file(uploaded_file):
    """Membaca isi dari file yang diupload (txt, docx, pdf)"""
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
            if pdfplumber is None:
                st.warning("⚠️ Library pdfplumber tidak ditemukan. Tidak dapat membaca file PDF.")
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


# --- FUNGSI UTAMA APLIKASI STREAMLIT ---
def main_app():
    """Fungsi utama untuk menjalankan aplikasi Streamlit"""
    st.set_page_config(
        page_title="NLP Toolkit: Representasi & Deep Learning",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 NLP TOOLKIT: Representasi Teks & Deep Learning")
    st.markdown("Aplikasi demonstrasi untuk preprocessing, representasi teks, dan analisis data teks.")

    # --- SIDEBAR INPUT ---
    with st.sidebar:
        st.header("⚙️ Konfigurasi Input")
        
        input_type = st.radio(
            "Pilih Tipe Input:",
            ("Input Manual", "Upload File", "X/Twitter Crawling (Simulasi)", "Web Scraping"),
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
                "Masukkan satu atau lebih dokumen (pisahkan dengan baris baru ganda):",
                "Jakarta is the capital city of Indonesia. I love this city.\n\n"
                "Natural Language Processing is a field of artificial intelligence.",
                key="manual_input_text"
            )
            if manual_text:
                list_dokumen_mentah = [doc.strip() for doc in manual_text.split('\n\n') if doc.strip()]
        
        elif input_type == "Upload File":
            st.subheader("📁 Upload Dokumen")
            st.info("Gunakan opsi ini untuk upload file (.csv, .txt, .docx, .pdf).")
            uploaded_file = st.file_uploader(
                "Upload file",
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

        elif input_type == "X/Twitter Crawling (Simulasi)":
            st.subheader("🔗 X/Twitter Crawling (Akses Token)")
            st.markdown("Masukkan token untuk mensimulasikan akses ke konten X/Twitter (tanpa keyword).")
            
            # --- INPUT TOKEN & PARAMETER CRAWLING ---
            auth_token_input = st.text_input(
                "Masukkan X/Twitter Auth Token Anda:",
                type="password",
                key="auth_token_input"
            )
            
            limit = st.slider("Jumlah Konten yang Diambil (Limit)", 10, 500, 50)
            
            # Penggabungan tombol execute dan analyze
            if st.button("🚀 Mulai Akses, Download & Analisis (Simulasi)"):
                if not auth_token_input:
                    st.error("❌ Auth Token diperlukan untuk memulai simulasi akses konten.")
                else:
                    st.info(f"🔄 Mensimulasikan akses ke konten penuh X/Twitter dengan Limit: {limit} data.")
                    
                    # Panggil fungsi simulasi tanpa keyword
                    df_hasil_crawl = simulate_x_crawling(limit)
                    
                    if not df_hasil_crawl.empty:
                        list_dokumen_mentah = df_hasil_crawl['full_text'].astype(str).dropna().tolist()
                        st.session_state.list_dokumen_mentah = list_dokumen_mentah
                        st.session_state.df_hasil_crawl = df_hasil_crawl
                        st.session_state.list_dokumen_bersih = []
                        
                        st.success(f"✅ Simulasi Akses Konten Selesai! Ditemukan {len(list_dokumen_mentah)} tweet. Lanjutkan ke tab Preprocessing.")
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
            url = st.text_input("Masukkan URL:", "https://en.wikipedia.org/wiki/Natural_language_processing", key="url_input")
            
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
                        else:
                            list_dokumen_mentah = [scraped_text]
                            st.session_state.list_dokumen_mentah = list_dokumen_mentah
                            st.session_state.list_dokumen_bersih = []
                            st.success("✅ Web Scraping selesai.")
                            st.text_area("Konten yang di-Scrape (Snippet):", scraped_text[:1000] + "...", height=200)

                    except Exception as e:
                        st.error(f"❌ Gagal melakukan web scraping: {e}")

        
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

    # --- TAB PREPROCESSING ---
    with tab_preprocess:
        st.header("1️⃣ PREPROCESSING DOKUMEN")
        list_dokumen_mentah_current = st.session_state.get('list_dokumen_mentah', [])
        
        if list_dokumen_mentah_current:
            if st.button("✨ Mulai Preprocessing (Sastrawi)", type="primary"):
                list_dokumen_bersih = []
                with st.spinner("Memproses semua dokumen..."):
                    for idx, doc in enumerate(list_dokumen_mentah_current):
                        st.subheader(f"Dokumen {idx+1} (Panjang: {len(doc.split())} kata)")
                        with st.expander("Lihat Teks Mentah", expanded=False):
                            st.code(doc[:2000], language="markdown") 
                        
                        cleaned_doc = preprocess_text(doc)
                        
                        if cleaned_doc:
                            list_dokumen_bersih.append(cleaned_doc)
                            st.markdown(f"**Hasil Preprocessing (Bersih):** (Panjang: {len(cleaned_doc.split())} kata)")
                            st.code(cleaned_doc[:2000], language="text") 
                        
                st.session_state.list_dokumen_bersih = list_dokumen_bersih
                st.success("✅ Preprocessing Selesai untuk semua dokumen!")
            else:
                if 'list_dokumen_bersih' not in st.session_state:
                     st.session_state.list_dokumen_bersih = []
                st.info("Klik tombol **'Mulai Preprocessing'** untuk memproses dokumen dan melanjutkan ke tab berikutnya.")
        else:
            st.warning("Input dokumen terlebih dahulu di sidebar.")

    # --- TAB REPRESENTASI TEKS ---
    with tab_repr:
        list_dokumen_bersih = st.session_state.get('list_dokumen_bersih', [])
        if list_dokumen_bersih:
            run_analysis(list_dokumen_bersih)
        else:
            st.warning("Lakukan Preprocessing terlebih dahulu di tab '1. Preprocessing'.")

    # --- TAB DEEP LEARNING ---
    with tab_dl:
        if DEEP_LEARNING_AVAILABLE:
            st.info("💡 **Catatan:** Klasifikasi Deep Learning menggunakan data dummy (sentimen positif/negatif) untuk demonstrasi.")
            
            if st.button("▶️ Jalankan Deep Learning Demo (Data Dummy)", type="primary", key="dl_demo_button"):
                dummy_texts, dummy_labels = generate_dummy_data()
                deep_learning_classification(dummy_texts, dummy_labels)
            
            st.markdown("---")
            st.subheader("Prediksi Teks Baru (Menggunakan Model Demo)")
            
            if 'trained_model' in st.session_state:
                # ... (Kode Prediksi) ...
                st.success("✅ Model Demo sudah dilatih dan siap digunakan.")
                new_text_predict = st.text_area("Masukkan teks baru untuk diprediksi sentimennya:", "This service was disappointing, not worth the price.")
                
                if st.button("🔮 Prediksi Teks Baru", key="predict_new_text"):
                    if new_text_predict.strip():
                        new_seq = st.session_state.tokenizer.texts_to_sequences([new_text_predict])
                        new_padded = pad_sequences(new_seq, maxlen=st.session_state.max_len, padding='post', truncating='post')
                        
                        prediction = st.session_state.trained_model.predict(new_padded, verbose=0)
                        
                        pred_label = int(prediction[0] > 0.5)
                        confidence = float(prediction[0]) if pred_label == 1 else 1 - float(prediction[0])
                        label_name = "Positive (1)" if pred_label == 1 else "Negative (0)"
                        
                        st.success(f"**Prediksi Sentimen:** {label_name}")
                        st.info(f"**Confidence:** {confidence:.2%}")
                    else:
                        st.warning("⚠️ Masukkan teks terlebih dahulu!")
            else:
                st.warning("Mohon jalankan **'Deep Learning Demo'** terlebih dahulu.")
        else:
            st.error("❌ Deep Learning (Tensorflow/Keras) tidak dapat dimuat.")


    # --- TAB ANALISIS LANJUTAN ---
    with tab_advanced:
        list_dokumen_mentah_current = st.session_state.get('list_dokumen_mentah', [])
        
        if not list_dokumen_mentah_current:
            st.warning("Input dokumen terlebih dahulu di sidebar.")
            return
            
        st.info("💡 **Catatan:** Analisis lanjutan ini lebih efektif pada **teks mentah** dan didukung untuk **Bahasa Inggris** oleh NLTK.")
        
        # POS Tagging
        st.markdown("---")
        pos_tagging_analysis(list_dokumen_mentah_current)
        
        # Named Entity Recognition
        st.markdown("---")
        named_entity_recognition(list_dokumen_mentah_current)


# --- RUN APP ---
if __name__ == '__main__':
    # Inisialisasi session state
    if 'list_dokumen_bersih' not in st.session_state:
        st.session_state.list_dokumen_bersih = []
    if 'list_dokumen_mentah' not in st.session_state:
        st.session_state.list_dokumen_mentah = []
    if 'last_input_type' not in st.session_state:
        st.session_state.last_input_type = 'Input Manual'
    
    main_app()
