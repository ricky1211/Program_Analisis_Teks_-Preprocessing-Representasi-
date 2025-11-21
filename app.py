# --- IMPORT LIBRARY ---
import re
import pandas as pd
import numpy as np
import os
import tempfile

# Web Framework
import streamlit as st

# Preprocessing
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Representasi Teks
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Document Processing
import docx
import pdfplumber

# Web Scraping
import requests
from bs4 import BeautifulSoup

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Speech Recognition
SPEECH_RECOGNITION_AVAILABLE = False
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    pass

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
        'punkt_tab',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
        'maxent_ne_chunker',
        'maxent_ne_chunker_tab',
        'words'
    ]
    
    for package in nltk_packages:
        try:
            # Cek di path standard NLTK
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                # Coba download jika tidak ada
                nltk.download(package, quiet=True)
            except:
                pass
        
        try:
            nltk.data.find(f'taggers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass
        
        try:
            nltk.data.find(f'chunkers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass
        
        try:
            nltk.data.find(f'corpora/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass

# Download data NLTK
try:
    download_nltk_data()
except Exception as e:
    st.warning(f"⚠️ Beberapa data NLTK gagal didownload: {e}")


# --- INISIALISASI MODEL ---
@st.cache_resource
def inisialisasi_model():
    """Menginisialisasi model Sastrawi"""
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    return stemmer, stopword_remover

try:
    stemmer, stopword_remover = inisialisasi_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model Sastrawi: {e}")
    # Jika gagal, hentikan aplikasi (penting untuk preprocessing)
    st.stop()


# --- FUNGSI PREPROCESSING ---
def tokenize_manual(text):
    """Tokenisasi manual menggunakan regex"""
    # Mencari kata yang hanya terdiri dari huruf
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    return tokens

def preprocess_text(text):
    """Fungsi preprocessing teks"""
    try:
        st.info("   📝 Case folding & cleaning...")
        text = text.lower()
        # Hapus karakter selain huruf dan spasi
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
        # Stemming hanya pada token yang tersisa
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
        st.info("💡 **Penjelasan:** Memberi bobot kata berdasarkan frekuensi (TF) dan kelangkaannya (IDF) di seluruh dokumen. Nilai yang lebih tinggi menandakan kata tersebut penting dan spesifik untuk dokumen tersebut.")
        
        max_tfidf = df_tfidf.max().sort_values(ascending=False).head(5)
        st.write("**Top 5 Kata Berdasarkan Bobot TF-IDF Tertinggi:**")
        st.bar_chart(max_tfidf)
        
    except ValueError as e:
        st.error(f"❌ ERROR TF-IDF: {e}")

    st.markdown("---")

    # WORD2VEC
    st.subheader("3️⃣ Metode Word2Vec")
    try:
        # Word2Vec membutuhkan list of list of tokens
        tokenized_docs_w2v = [doc.split() for doc in list_dokumen_bersih if doc]
        
        if not tokenized_docs_w2v:
            st.error("❌ ERROR: Tidak ada token untuk dilatih.")
            return

        st.write("**Input untuk Word2Vec (Sample):**")
        # Ambil maksimal 50 kata pertama dari 2 dokumen
        sample_code = [doc[:50] for doc in tokenized_docs_w2v[:2]]
        st.code(str(sample_code).replace('\'', '"'), language="python")
        
        with st.spinner("🔄 Melatih model Word2Vec (Vector Size=100, Window=5)..."):
            model_w2v = Word2Vec(
                sentences=tokenized_docs_w2v, 
                vector_size=100, # Ukuran vektor
                window=5,        # Jarak maksimum antara kata
                min_count=1,     # Abaikan kata dengan frekuensi kurang dari ini
                workers=4,
                epochs=10
            )
        
        st.success("✅ Model Word2Vec berhasil dilatih!")
        
        vocab_size = len(model_w2v.wv.index_to_key)
        st.metric("Ukuran Vocabulary", vocab_size)
        
        if model_w2v.wv.index_to_key:
            kata_uji = model_w2v.wv.index_to_key[0] # Ambil kata pertama di vocabulary
            
            st.write(f"**Contoh Vektor untuk kata '{kata_uji}'** (10 dimensi pertama):")
            # Word2Vec menghasilkan vektor numerik (Embedding)
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
    # 1 = Positive, 0 = Negative
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    return texts, np.array(labels)


def deep_learning_classification(texts, labels):
    """
    Klasifikasi teks menggunakan RNN/LSTM
    
    Parameters:
    - texts: list of strings (dokumen teks)
    - labels: numpy array of integers (label klasifikasi)
    """
    st.header("🤖 DEEP LEARNING - RNN/LSTM CLASSIFICATION")
    
    st.info("""
    **Deep Learning untuk Klasifikasi Teks:**
    - Menggunakan arsitektur RNN (Simple RNN, LSTM, Bidirectional LSTM)
    - Model memetakan kata menjadi vektor (Embedding) dan mempelajari urutan kata.
    - Cocok untuk klasifikasi sentimen (demonstrasi ini), kategori dokumen, dll.
    """)
    
    # Validasi input
    if len(texts) < 10:
        st.warning("⚠️ Data terlalu sedikit untuk training.")
        return
    
    # PREPROCESSING UNTUK DEEP LEARNING
    st.subheader("1️⃣ Preprocessing Data")
    
    with st.spinner("Memproses teks..."):
        # Parameter
        max_words = 5000  # Jumlah kata maksimum yang akan disimpan di vocabulary
        max_len = 100     # Panjang maksimum sequence (dokumen)
        
        # Tokenisasi (Keras Tokenizer)
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        # Padding Sequence
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        # Info preprocessing
        vocab_size = len(tokenizer.word_index) + 1
        st.success(f"✅ Preprocessing selesai!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vocabulary Size", vocab_size)
        with col2:
            st.metric("Max Sequence Length", max_len)
        with col3:
            st.metric("Total Samples", len(texts))
    
    st.markdown("---")
    
    # SPLIT DATA
    st.subheader("2️⃣ Split Data (Train/Test)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Testing Samples", len(X_test))
    
    st.markdown("---")
    
    # PILIH ARSITEKTUR MODEL
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
        
        # Embedding Layer
        embedding_dim = 128
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        
        # Recurrent Layer
        if model_type == "Simple RNN":
            model.add(SimpleRNN(64, return_sequences=False))
        elif model_type == "LSTM":
            model.add(LSTM(64, return_sequences=False))
        elif model_type == "Bidirectional LSTM":
            # Bidirectional LSTM dapat menangkap konteks maju dan mundur
            model.add(Bidirectional(LSTM(64, return_sequences=False))) 
        
        model.add(Dropout(0.5)) # Regularisasi
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        
        # Output Layer (Binary classification - 2 classes: Positive/Negative)
        num_classes = len(np.unique(labels))
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss_fn = 'binary_crossentropy'
        else:
            # Multi-class classification
            model.add(Dense(num_classes, activation='softmax'))
            loss_fn = 'sparse_categorical_crossentropy'
        
        # Compile Model
        model.compile(
            optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy']
        )
    
    st.success(f"✅ Model {model_type} berhasil dibuat!")
    
    # Tampilkan arsitektur model
    with st.expander("📋 Lihat Arsitektur Model"):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.code('\n'.join(model_summary))
    
    st.markdown("---")
    
    # TRAINING MODEL
    st.subheader("4️⃣ Training Model dengan Backpropagation")
    
    if st.button("🚀 Mulai Training", type="primary", key="start_training_button"):
        with st.spinner(f"Training model selama {epochs} epochs..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Custom callback untuk update progress
            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f} - Acc: {logs['accuracy']:.4f} - Val Loss: {logs['val_loss']:.4f} - Val Acc: {logs['val_accuracy']:.4f}")
            
            # Training
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
        
        # Plot Training History
        st.subheader("📈 Training History")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss Plot
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy Plot
        ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # EVALUASI MODEL
        st.subheader("5️⃣ Evaluasi Model")
        
        # Prediksi
        y_pred_proba = model.predict(X_test, verbose=0)
        
        if num_classes == 2:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        # Menggunakan average='binary' untuk 2 kelas (karena data dummy)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1) 
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("📊 F1-Score (Positive Class)", f"{f1:.4f}")
        with col3:
            final_loss = history.history['val_loss'][-1]
            st.metric("📉 Validation Loss", f"{final_loss:.4f}")
        
        # Classification Report
        st.write("**📋 Classification Report:**")
        # Target names untuk data dummy: 0=Negative, 1=Positive
        target_names = ['Negative (0)', 'Positive (1)']
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.round(4), use_container_width=True)
        
        # Confusion Matrix
        st.write("**🔀 Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Label untuk Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=target_names, yticklabels=target_names)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)
        
        # Simpan model ke session state
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
                # Tokenisasi
                tokens = word_tokenize(doc.lower())
                
                if not tokens:
                    st.warning("⚠️ Tidak ada token yang dihasilkan")
                    continue
                
                # POS Tagging
                pos_tags = pos_tag(tokens)
                
                # Tampilkan dalam tabel
                df_pos = pd.DataFrame(pos_tags, columns=['Word', 'POS Tag'])
                st.dataframe(df_pos, use_container_width=True, height=300)
                
                # Statistik POS Tag
                pos_counts = df_pos['POS Tag'].value_counts()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Distribusi Top 10 POS Tags:**")
                    st.bar_chart(pos_counts.head(10))
                
                with col2:
                    st.write("**Top 5 POS Tags:**")
                    for pos, count in pos_counts.head(5).items():
                        st.metric(pos, count)
                
                # Penjelasan minimal 3 POS Tag menarik
                st.write("**📚 Penjelasan POS Tags yang Menarik (dari NLTK):**")
                
                pos_explanations = {
                    'NN': '**NN (Noun, singular)**: Kata benda tunggal (ex: table, car)',
                    'NNS': '**NNS (Noun, plural)**: Kata benda jamak (ex: tables, cars)',
                    'VB': '**VB (Verb, base form)**: Kata kerja bentuk dasar (ex: eat, go)',
                    'VBD': '**VBD (Verb, past tense)**: Kata kerja bentuk lampau (ex: ate, went)',
                    'JJ': '**JJ (Adjective)**: Kata sifat (ex: good, big)',
                    'RB': '**RB (Adverb)**: Kata keterangan (ex: very, quickly)',
                    'PRP': '**PRP (Personal Pronoun)**: Kata ganti personal (ex: I, he, she)',
                    'DT': '**DT (Determiner)**: Kata penentu (ex: the, a, an)',
                    'IN': '**IN (Preposition/subordinating conjunction)**: Kata depan (ex: in, of, on)'
                }
                
                explained_tags = []
                for pos in pos_counts.head(5).index:
                    if pos in pos_explanations:
                        st.markdown(f"- {pos_explanations[pos]}")
                        explained_tags.append(pos)
                
            except LookupError as e:
                st.error(f"❌ Error NLTK: {e}")
                st.warning("⚠️ Pastikan data NLTK **'punkt'** dan **'averaged\_perceptron\_tagger'** sudah didownload.")
            except Exception as e:
                st.error(f"❌ Error POS Tagging: {e}")


def named_entity_recognition(list_dokumen_mentah):
    """B - Named Entity Recognition: Identifikasi entitas bernama"""
    st.header("🎯 B - NAMED ENTITY RECOGNITION (NER)")
    st.info("💡 **NER** mengidentifikasi dan mengklasifikasikan entitas bernama seperti nama orang, lokasi, organisasi, tanggal, dll. ")
    
    for idx, doc in enumerate(list_dokumen_mentah):
        with st.expander(f"📄 NER - Dokumen {idx+1}", expanded=(idx==0)):
            try:
                # Tokenisasi dan POS Tagging
                tokens = word_tokenize(doc)
                pos_tags = pos_tag(tokens)
                
                # Named Entity Recognition
                # binary=False menggunakan semua tipe NE (PERSON, GPE, ORGANIZATION, dll)
                named_entities = ne_chunk(pos_tags, binary=False) 
                
                # Ekstrak entitas
                entities_dict = {
                    'PERSON': [],
                    'LOCATION': [],
                    'ORGANIZATION': [],
                    'DATE': [],
                    'TIME': [],
                    'MONEY': [],
                    'GPE': [],  # Geopolitical Entity (Country, City, State)
                    'OTHER': []
                }
                
                for chunk in named_entities:
                    if hasattr(chunk, 'label'):
                        # Gabungkan kata-kata dalam entitas
                        entity_text = ' '.join(c[0] for c in chunk)
                        entity_label = chunk.label()
                        
                        if entity_label in entities_dict:
                            entities_dict[entity_label].append(entity_text)
                        else:
                            entities_dict['OTHER'].append(f"{entity_text} ({entity_label})")
                
                # Tampilkan entitas yang ditemukan
                st.write("**🔍 Entitas yang Ditemukan:**")
                
                found_entities = False
                for category, entities in entities_dict.items():
                    if entities:
                        found_entities = True
                        with st.container(border=True):
                            st.subheader(category)
                            # Hapus duplikat
                            unique_entities = sorted(list(set(entities)))
                            st.markdown("- " + "\n- ".join(unique_entities))
                
                if not found_entities:
                    st.warning("⚠️ Tidak ada entitas bernama yang ditemukan dalam dokumen ini.")
                    st.info("💡 Tip: Pastikan teks mengandung nama orang, tempat, atau organisasi dalam bahasa Inggris untuk deteksi optimal.")
                
                st.markdown("---")
                
                # Penjelasan Kategori
                st.write("**📚 Penjelasan Kategori Entitas:**")
                st.markdown("""
                - **PERSON**: Nama orang (contoh: John Doe)
                - **GPE**: Entitas Geopolitical (contoh: Indonesia, Jakarta)
                - **ORGANIZATION**: Organisasi, perusahaan (contoh: Google, UN)
                - **LOCATION**: Lokasi geografis non-GPE (contoh: Mount Everest, Pacific Ocean)
                - **DATE/TIME**: Ekspresi temporal (contoh: 2023, yesterday)
                - **MONEY**: Nilai mata uang
                """)

                # Tampilkan Struktur Pohon (Visualisasi NER)
                st.write("**🌳 Struktur Pohon NER (Chunked Tree):**")
                
                # Fungsi untuk menampilkan tree (menggunakan representasi string)
                def tree_to_string(tree):
                    # Menggunakan metode str() bawaan Tree
                    return str(tree)
                
                tree_string = tree_to_string(named_entities)
                # Tampilkan 20 baris pertama
                st.code('\n'.join(tree_string.splitlines()[:20]), language="text")

                st.info("💡 **Penjelasan Struktur Pohon:** Entitas bernama dikelompokkan (chunking) dan diberi label di atas kata-kata yang membentuknya.")
                
            except LookupError as e:
                st.error(f"❌ Error NLTK NER: {e}")
                st.warning("⚠️ Pastikan data NLTK **'maxent_ne_chunker'** dan **'words'** sudah didownload.")
            except Exception as e:
                st.error(f"❌ Error NER: {e}")


# --- FUNGSI UTAMA UNTUK MEMBACA DOKUMEN ---
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
            # Menggunakan tempfile untuk memastikan pdfplumber bisa mengakses file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        text_content += page.extract_text() if page.extract_text() else ""
            finally:
                os.remove(tmp_path) # Pastikan file temporary dihapus
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
    st.markdown("Aplikasi demonstrasi untuk preprocessing, representasi teks (BoW, TF-IDF, Word2Vec), klasifikasi Deep Learning (RNN/LSTM), dan analisis lanjutan (POS Tagging, NER).")

    # --- SIDEBAR INPUT ---
    with st.sidebar:
        st.header("⚙️ Konfigurasi Input")
        
        # Pilihan Input Teks
        input_type = st.radio(
            "Pilih Tipe Input:",
            ("Input Manual", "Upload File", "Web Scraping"),
            index=0
        )
        
        list_dokumen_mentah = []
        
        if input_type == "Input Manual":
            st.subheader("📝 Input Teks Manual")
            manual_text = st.text_area(
                "Masukkan satu atau lebih dokumen (pisahkan dengan baris baru ganda, gunakan bahasa Inggris untuk hasil NER/POS optimal):",
                "Jakarta is the capital city of Indonesia. I love this city.\n\n"
                "Natural Language Processing (NLP) is a field of artificial intelligence."
            )
            if manual_text:
                # Pisahkan dokumen berdasarkan baris baru ganda
                list_dokumen_mentah = [doc.strip() for doc in manual_text.split('\n\n') if doc.strip()]
        
        elif input_type == "Upload File":
            st.subheader("📁 Upload Dokumen")
            uploaded_file = st.file_uploader(
                "Upload file (.txt, .docx, .pdf)",
                type=["txt", "docx", "pdf"],
                accept_multiple_files=True
            )
            
            if uploaded_file:
                for file in uploaded_file:
                    content = read_uploaded_file(file)
                    if content:
                        list_dokumen_mentah.append(content)
                if list_dokumen_mentah:
                    st.success(f"✅ Berhasil membaca {len(list_dokumen_mentah)} dokumen.")

        elif input_type == "Web Scraping":
            st.subheader("🔗 Web Scraping")
            url = st.text_input("Masukkan URL (Contoh: https://en.wikipedia.org/wiki/NLP):", 
                                "https://en.wikipedia.org/wiki/Natural_language_processing")
            
            if st.button("Scrape Konten"):
                if url:
                    try:
                        st.info(f"🔄 Mengambil konten dari: **{url}**")
                        response = requests.get(url, timeout=10)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Ekstrak semua teks dari tag <p>
                        paragraphs = soup.find_all('p')
                        scraped_text = '\n'.join([p.get_text() for p in paragraphs])
                        
                        if not scraped_text.strip():
                            st.warning("⚠️ Tidak ada konten paragraf yang ditemukan.")
                        else:
                            # Anggap seluruh konten sebagai satu dokumen
                            list_dokumen_mentah.append(scraped_text)
                            st.session_state.list_dokumen_mentah_new = list_dokumen_mentah # Trigger update
                            st.success("✅ Web Scraping selesai.")
                            st.text_area("Konten yang di-Scrape (Snippet):", scraped_text[:1000] + "...", height=200)

                    except requests.exceptions.Timeout:
                        st.error("❌ Timeout: Permintaan melebihi batas waktu.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Gagal koneksi ke URL: {e}")
                    except Exception as e:
                        st.error(f"❌ Gagal melakukan web scraping: {e}")
        
        # Load dokumen mentah dari session state jika ada
        if 'list_dokumen_mentah_new' in st.session_state and input_type != "Web Scraping":
            list_dokumen_mentah = st.session_state.list_dokumen_mentah_new
        
        # Simpan kembali ke session state untuk dibaca tab lain
        st.session_state.list_dokumen_mentah = list_dokumen_mentah

        st.markdown("---")
        st.subheader("🗂️ Status Dokumen")
        st.metric("Jumlah Dokumen Mentah", len(list_dokumen_mentah))
        
        if len(list_dokumen_mentah) == 0:
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
        
        if list_dokumen_mentah:
            if st.button("✨ Mulai Preprocessing (Sastrawi)", type="primary"):
                list_dokumen_bersih = []
                with st.spinner("Memproses semua dokumen..."):
                    for idx, doc in enumerate(list_dokumen_mentah):
                        st.subheader(f"Dokumen {idx+1}")
                        with st.expander("Lihat Teks Mentah", expanded=False):
                            st.code(doc, language="markdown")
                        
                        # Jalankan preprocessing
                        cleaned_doc = preprocess_text(doc)
                        
                        if cleaned_doc:
                            list_dokumen_bersih.append(cleaned_doc)
                            st.markdown("**Hasil Preprocessing (Bersih):**")
                            st.code(cleaned_doc, language="text")
                        
                st.session_state.list_dokumen_bersih = list_dokumen_bersih
                st.success("✅ Preprocessing Selesai untuk semua dokumen!")
            else:
                # Muat dari session state jika sudah diproses
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
        st.info("💡 **Catatan:** Klasifikasi Deep Learning membutuhkan data berlabel. Kami menggunakan **data dummy (sentimen positif/negatif)** untuk mendemonstrasikan proses training dan evaluasi.")
        
        # Tombol untuk menjalankan Deep Learning dengan data dummy
        if st.button("▶️ Jalankan Deep Learning Demo (Data Dummy)", type="primary", key="dl_demo_button"):
            dummy_texts, dummy_labels = generate_dummy_data()
            deep_learning_classification(dummy_texts, dummy_labels)
        
        st.markdown("---")
        st.subheader("Prediksi Teks Baru (Menggunakan Model Demo)")
        
        if 'trained_model' in st.session_state:
            model = st.session_state.trained_model
            tokenizer = st.session_state.tokenizer
            max_len = st.session_state.max_len
            
            st.success("✅ Model Demo (Simple RNN/LSTM) sudah dilatih dan siap digunakan.")
            
            new_text_predict = st.text_area("Masukkan teks baru untuk diprediksi sentimennya (Contoh: I love this, it is bad):", 
                                            "This service was disappointing, not worth the price.")
            
            if st.button("🔮 Prediksi Teks Baru", key="predict_new_text"):
                if new_text_predict.strip():
                    # Tokenize and Pad
                    new_seq = tokenizer.texts_to_sequences([new_text_predict])
                    new_padded = pad_sequences(new_seq, maxlen=max_len, padding='post', truncating='post')
                    
                    # Predict
                    prediction = model.predict(new_padded, verbose=0)
                    
                    # Asumsi 2 kelas (karena data dummy)
                    pred_label = int(prediction[0] > 0.5)
                    confidence = float(prediction[0]) if pred_label == 1 else 1 - float(prediction[0])
                    label_name = "Positive (1)" if pred_label == 1 else "Negative (0)"
                    
                    st.success(f"**Prediksi Sentimen:** {label_name}")
                    st.info(f"**Confidence:** {confidence:.2%}")
                    
                    st.progress(confidence)
                else:
                    st.warning("⚠️ Masukkan teks terlebih dahulu!")
            
        else:
            st.warning("Mohon jalankan **'Deep Learning Demo'** terlebih dahulu.")

    # --- TAB ANALISIS LANJUTAN ---
    with tab_advanced:
        list_dokumen_mentah = st.session_state.get('list_dokumen_mentah', [])
        
        if not list_dokumen_mentah:
            st.warning("Input dokumen terlebih dahulu di sidebar.")
            return
            
        st.info("💡 **Catatan:** Analisis lanjutan (POS/NER) ini lebih efektif pada **teks mentah (belum di-stemming)** dan terutama didukung untuk **Bahasa Inggris** oleh library NLTK.")
        
        # POS Tagging
        st.markdown("---")
        pos_tagging_analysis(list_dokumen_mentah)
        
        # Named Entity Recognition
        st.markdown("---")
        named_entity_recognition(list_dokumen_mentah)


# --- RUN APP ---
if __name__ == '__main__':
    # Inisialisasi session state
    if 'list_dokumen_bersih' not in st.session_state:
        st.session_state.list_dokumen_bersih = []
    if 'list_dokumen_mentah' not in st.session_state:
        st.session_state.list_dokumen_mentah = []
    
    # Run main application
    main_app()
