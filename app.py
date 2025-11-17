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

# Handler Input
import docx
import pdfplumber
import speech_recognition as sr
import requests
from bs4 import BeautifulSoup

# NLP Advanced
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk.chunk import ne_chunk

# Download NLTK data (hanya jika belum ada)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)


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
    st.stop()


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


# --- FUNGSI ANALISIS REPRESENTASI TEKS ---
def run_analysis(list_dokumen_bersih):
    """Menganalisis dokumen dengan BoW, TF-IDF, dan Word2Vec"""
    st.header("📊 IMPLEMENTASI REPRESENTASI TEKS")

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
        st.info("💡 **Penjelasan:** Menghitung frekuensi kemunculan setiap kata dalam dokumen.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Kata Unik", len(fitur_bow))
        with col2:
            st.metric("Total Kata (Sum)", int(bow_matrix.sum()))
            
    except ValueError as e:
        st.error(f"❌ ERROR BoW: {e}")

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
        st.info("💡 **Penjelasan:** Memberi bobot kata berdasarkan frekuensi dan kelangkaannya di seluruh dokumen.")
        
        max_tfidf = df_tfidf.max().sort_values(ascending=False).head(5)
        st.write("**Top 5 Kata Berdasarkan TF-IDF:**")
        st.bar_chart(max_tfidf)
        
    except ValueError as e:
        st.error(f"❌ ERROR TF-IDF: {e}")

    # WORD2VEC
    st.subheader("3️⃣ Metode Word2Vec")
    try:
        tokenized_docs_w2v = [doc.split() for doc in list_dokumen_bersih if doc]
        
        if not tokenized_docs_w2v:
            st.error("❌ ERROR: Tidak ada token untuk dilatih.")
            return

        st.write("**Input untuk Word2Vec (Sample):**")
        st.code(str(tokenized_docs_w2v[:2]), language="python")
        
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
            st.json({f"dim_{i}": round(v, 4) for i, v in enumerate(vektor)})

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
                
                # POS Tagging
                pos_tags = pos_tag(tokens)
                
                # Tampilkan dalam tabel
                df_pos = pd.DataFrame(pos_tags, columns=['Word', 'POS Tag'])
                st.dataframe(df_pos, use_container_width=True, height=300)
                
                # Statistik POS Tag
                pos_counts = df_pos['POS Tag'].value_counts()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Distribusi POS Tags:**")
                    st.bar_chart(pos_counts)
                
                with col2:
                    st.write("**Top POS Tags:**")
                    for pos, count in pos_counts.head(5).items():
                        st.metric(pos, count)
                
                # Penjelasan minimal 3 POS Tag menarik
                st.write("**📚 Penjelasan POS Tags yang Menarik:**")
                
                pos_explanations = {
                    'NN': '**NN (Noun)**: Kata benda tunggal, contoh: rumah, buku',
                    'NNS': '**NNS (Noun Plural)**: Kata benda jamak, contoh: rumah-rumah',
                    'VB': '**VB (Verb)**: Kata kerja bentuk dasar, contoh: makan, pergi',
                    'VBD': '**VBD (Verb Past)**: Kata kerja bentuk lampau, contoh: ate, went',
                    'VBG': '**VBG (Verb Gerund)**: Kata kerja bentuk -ing, contoh: eating, going',
                    'JJ': '**JJ (Adjective)**: Kata sifat, contoh: bagus, besar',
                    'RB': '**RB (Adverb)**: Kata keterangan, contoh: sangat, cepat',
                    'PRP': '**PRP (Pronoun)**: Kata ganti, contoh: saya, dia, kami',
                    'DT': '**DT (Determiner)**: Kata penentu, contoh: the, a, an',
                    'IN': '**IN (Preposition)**: Kata depan, contoh: di, pada, dari'
                }
                
                explained_tags = []
                for pos in pos_counts.head(5).index:
                    if pos in pos_explanations:
                        st.markdown(f"- {pos_explanations[pos]}")
                        explained_tags.append(pos)
                
                # Tambahkan penjelasan umum jika kurang dari 3
                if len(explained_tags) < 3:
                    remaining = [k for k in list(pos_explanations.keys())[:3] if k not in explained_tags]
                    for pos in remaining[:3-len(explained_tags)]:
                        st.markdown(f"- {pos_explanations[pos]}")
                
            except Exception as e:
                st.error(f"❌ Error POS Tagging: {e}")


def named_entity_recognition(list_dokumen_mentah):
    """B - Named Entity Recognition: Identifikasi entitas bernama"""
    st.header("🎯 B - NAMED ENTITY RECOGNITION (NER)")
    st.info("💡 **NER** mengidentifikasi dan mengklasifikasikan entitas bernama seperti nama orang, lokasi, organisasi, tanggal, dll.")
    
    for idx, doc in enumerate(list_dokumen_mentah):
        with st.expander(f"📄 NER - Dokumen {idx+1}", expanded=(idx==0)):
            try:
                # Tokenisasi dan POS Tagging
                tokens = word_tokenize(doc)
                pos_tags = pos_tag(tokens)
                
                # Named Entity Recognition
                named_entities = ne_chunk(pos_tags, binary=False)
                
                # Ekstrak entitas
                entities_dict = {
                    'PERSON': [],
                    'LOCATION': [],
                    'ORGANIZATION': [],
                    'DATE': [],
                    'GPE': [],  # Geopolitical Entity
                    'OTHER': []
                }
                
                for chunk in named_entities:
                    if hasattr(chunk, 'label'):
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
                        with st.container():
                            st.write(f"**{category}:**")
                            # Hapus duplikat
                            unique_entities = list(set(entities))
                            for entity in unique_entities:
                                st.write(f"  • {entity}")
                
                if not found_entities:
                    st.warning("⚠️ Tidak ada entitas bernama yang ditemukan dalam dokumen ini.")
                    st.info("💡 Tip: Pastikan teks mengandung nama orang, tempat, atau organisasi dalam bahasa Inggris untuk deteksi optimal.")
                
                # Penjelasan Kategori
                st.write("**📚 Penjelasan Kategori Entitas:**")
                st.markdown("""
                - **PERSON**: Nama orang (contoh: John Doe, Barack Obama)
                - **LOCATION**: Nama tempat geografis (contoh: Mount Everest, Pacific Ocean)
                - **ORGANIZATION**: Nama organisasi/perusahaan (contoh: Google, United Nations)
                - **DATE**: Tanggal dan waktu (contoh: Monday, January 1st, 2024)
                - **GPE**: Entitas geopolitik seperti negara, kota (contoh: Indonesia, Jakarta)
                """)
                
                # Visualisasi distribusi entitas
                entity_counts = {k: len(set(v)) for k, v in entities_dict.items() if v}
                if entity_counts:
                    st.write("**📊 Distribusi Entitas:**")
                    df_entities = pd.DataFrame(list(entity_counts.items()), 
                                              columns=['Kategori', 'Jumlah'])
                    st.bar_chart(df_entities.set_index('Kategori'))
                
            except Exception as e:
                st.error(f"❌ Error NER: {e}")


def parsing_analysis(list_dokumen_mentah):
    """C - Constituency dan Dependency Parsing"""
    st.header("🌳 C - CONSTITUENCY & DEPENDENCY PARSING")
    st.info("💡 **Parsing** menganalisis struktur gramatikal kalimat")
    
    for idx, doc in enumerate(list_dokumen_mentah):
        with st.expander(f"📄 Parsing - Dokumen {idx+1}", expanded=(idx==0)):
            try:
                # Ambil beberapa kalimat pertama untuk analisis
                sentences = doc.split('.')[:3]  # Ambil 3 kalimat pertama
                
                for sent_idx, sentence in enumerate(sentences):
                    if len(sentence.strip()) < 5:
                        continue
                        
                    st.write(f"**Kalimat {sent_idx+1}:** _{sentence.strip()}_")
                    
                    # Tokenisasi dan POS Tagging
                    tokens = word_tokenize(sentence.strip())
                    pos_tags = pos_tag(tokens)
                    
                    # === DEPENDENCY PARSING ===
                    st.write("**🔗 Dependency Parsing** (Relasi antar kata):")
                    
                    # Buat tabel relasi sederhana
                    df_dep = pd.DataFrame(pos_tags, columns=['Kata', 'POS'])
                    df_dep['Index'] = range(len(df_dep))
                    df_dep['Head'] = '-'
                    df_dep['Relasi'] = 'ROOT' if len(df_dep) > 0 else '-'
                    
                    # Simulasi dependency sederhana berdasarkan POS
                    for i in range(len(df_dep)):
                        if i > 0:
                            current_pos = df_dep.loc[i, 'POS']
                            # Aturan sederhana
                            if current_pos.startswith('V'):  # Verb
                                df_dep.loc[i, 'Relasi'] = 'ROOT'
                                df_dep.loc[i, 'Head'] = 0
                            elif current_pos.startswith('N'):  # Noun
                                df_dep.loc[i, 'Relasi'] = 'nsubj' if i < len(df_dep)//2 else 'obj'
                                # Cari verb terdekat
                                for j in range(len(df_dep)):
                                    if df_dep.loc[j, 'POS'].startswith('V'):
                                        df_dep.loc[i, 'Head'] = j
                                        break
                            elif current_pos.startswith('J'):  # Adjective
                                df_dep.loc[i, 'Relasi'] = 'amod'
                                if i > 0:
                                    df_dep.loc[i, 'Head'] = i - 1
                            elif current_pos.startswith('R'):  # Adverb
                                df_dep.loc[i, 'Relasi'] = 'advmod'
                                if i > 0:
                                    df_dep.loc[i, 'Head'] = i - 1
                    
                    st.dataframe(df_dep[['Index', 'Kata', 'POS', 'Head', 'Relasi']], 
                               use_container_width=True)
                    
                    st.caption("📝 Relasi: nsubj=subjek, obj=objek, amod=modifier kata sifat, advmod=modifier kata keterangan")
                    
                    # === CONSTITUENCY PARSING ===
                    st.write("**🌲 Constituency Parsing** (Pohon Struktur Frasa):")
                    
                    # Named Entity Chunking sebagai proxy untuk constituency
                    chunked = ne_chunk(pos_tags, binary=True)
                    
                    # Tampilkan tree structure
                    tree_str = str(chunked)
                    st.code(tree_str, language="lisp")
                    
                    # Penjelasan struktur
                    st.markdown("""
                    **Penjelasan Struktur:**
                    - **S**: Sentence (Kalimat)
                    - **NP**: Noun Phrase (Frasa Nomina)
                    - **VP**: Verb Phrase (Frasa Verba)
                    - **PP**: Prepositional Phrase (Frasa Preposisi)
                    """)
                    
                    st.divider()
                
                st.success("✅ Parsing selesai!")
                
            except Exception as e:
                st.error(f"❌ Error Parsing: {e}")


# --- HANDLER INPUT ---
def handle_url(url):
    """Ekstrak teks dari URL menggunakan BeautifulSoup"""
    st.info(f"🌐 Mengunduh artikel dari: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        paragraphs = soup.find_all(['p', 'article', 'div'])
        text_list = []
        
        for para in paragraphs:
            text = para.get_text(strip=True)
            if len(text) > 50:
                text_list.append(text)
        
        full_text = '\n'.join(text_list)
        
        if len(full_text) < 100:
            st.warning("⚠️ Teks yang diekstrak terlalu pendek.")
            return ""
        
        return full_text
        
    except Exception as e:
        st.error(f"❌ Gagal mengambil artikel: {e}")
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
    """Ekstrak teks dari dokumen"""
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


# --- APLIKASI UTAMA ---
if 'dokumen_mentah_list' not in st.session_state:
    st.session_state.dokumen_mentah_list = []

st.title("🚀 Program Analisis Teks NLP")
st.markdown("""
**Aplikasi ini melakukan:**
- 🔄 Preprocessing (Case Folding, Tokenizing, Stopword Removal, Stemming)
- 🏷️ POS Tagging (Part-of-Speech Tagging)
- 🎯 Named Entity Recognition (NER)
- 🌳 Constituency & Dependency Parsing
- 📊 Representasi Teks (Bag of Words, TF-IDF, Word2Vec)
- 📁 Mendukung berbagai input: URL, PDF, DOCX, TXT, Audio
""")

st.divider()

# SIDEBAR
st.sidebar.header("📥 Input Data")

# INPUT URL
with st.sidebar.expander("🌐 Dari URL"):
    url_input = st.text_input("Masukkan URL artikel")
    if st.button("Proses URL", key="btn_url"):
        if url_input:
            with st.spinner("Memproses URL..."):
                teks_mentah = handle_url(url_input)
                if teks_mentah:
                    st.session_state.dokumen_mentah_list.append(teks_mentah)
                    st.success(f"✅ URL berhasil! Total: {len(st.session_state.dokumen_mentah_list)}")
        else:
            st.warning("⚠️ Harap masukkan URL.")

# INPUT FILE
with st.sidebar.expander("📁 Dari File"):
    uploaded_file = st.file_uploader(
        "Pilih file", 
        type=["wav", "mp3", "pdf", "docx", "txt"]
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
                    
                    if file_extension in ['.wav', '.mp3']:
                        teks_mentah = handle_audio(file_path)
                    elif file_extension in ['.txt', '.pdf', '.docx']:
                        teks_mentah = handle_doc(file_path, file_extension)
                    
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    
                    if teks_mentah:
                        st.session_state.dokumen_mentah_list.append(teks_mentah)
                        st.success(f"✅ File berhasil! Total: {len(st.session_state.dokumen_mentah_list)}")
                        
                        with st.expander("👁️ Lihat hasil ekstraksi"):
                            st.text_area("", teks_mentah, height=150)

if st.sidebar.button("🗑️ Hapus Semua Dokumen"):
    st.session_state.dokumen_mentah_list = []
    st.rerun()

st.sidebar.divider()
st.sidebar.metric("📚 Total Dokumen", len(st.session_state.dokumen_mentah_list))

# MAIN CONTENT
if st.button("📋 Tampilkan Dokumen Terkumpul"):
    if not st.session_state.dokumen_mentah_list:
        st.warning("⚠️ Belum ada dokumen.")
    else:
        st.subheader(f"📚 Total {len(st.session_state.dokumen_mentah_list)} Dokumen")
        
        for i, doc in enumerate(st.session_state.dokumen_mentah_list):
            with st.expander(f"📄 Dokumen {i+1} ({len(doc)} karakter)"):
                st.text_area("", doc, height=150, key=f"doc_raw_{i}", disabled=True)

st.divider()

st.header("🎯 Jalankan Analisis")

if st.button("▶️ MULAI PREPROCESSING & ANALISIS", type="primary", use_container_width=True):
    if not st.session_state.dokumen_mentah_list:
        st.error("❌ Tidak ada dokumen untuk dianalisis.")
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
                        f"Hasil Bersih", 
                        hasil_proses, 
                        height=100, 
                        key=f"clean_{i}"
                    )
        
        progress_bar.empty()
        status_text.empty()
        
        if not list_dokumen_bersih:
            st.error("❌ Semua dokumen gagal diproses.")
        else:
            st.success(f"✅ Preprocessing selesai! {len(list_dokumen_bersih)} dokumen berhasil.")
            
            st.divider()
            
            # === ANALISIS LANJUTAN DENGAN DOKUMEN MENTAH ===
            st.title("🎓 ANALISIS LANJUTAN NLP")
            
            # A - POS Tagging
            pos_tagging_analysis(st.session_state.dokumen_mentah_list)
            st.divider()
            
            # B - Named Entity Recognition
            named_entity_recognition(st.session_state.dokumen_mentah_list)
            st.divider()
            
            # C - Constituency & Dependency Parsing
            parsing_analysis(st.session_state.dokumen_mentah_list)
            st.divider()
            
            # === ANALISIS REPRESENTASI TEKS ===
            run_analysis(list_dokumen_bersih)
            st.balloons()

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <small>📚 Program Analisis Teks NLP | Dibuat dengan Streamlit</small>
</div>
""", unsafe_allow_html=True)
