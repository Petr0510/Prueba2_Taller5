import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline

# Descargar recursos necesarios
nltk.download("punkt")
nltk.download("stopwords")

# Configurar página
st.set_page_config(
    page_title="Análisis de Opiniones de Clientes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>📊 Análisis de Opiniones de Clientes</h1>
    <p style='text-align: center; color: #666;'>Explora sentimientos, palabras clave y temas recurrentes</p>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Cargando modelo de análisis de sentimiento...")
def get_sentiment_pipeline():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

sentiment_pipe = get_sentiment_pipeline()

# Clasificación de sentimiento
def get_sentiment(text):
    try:
        result = sentiment_pipe(text)
        label = result[0]['label'] if result else "Neutral"
        if label in ["Very Positive", "Positive"]:
            return "Positivo"
        elif label == "Neutral":
            return "Neutral"
        elif label in ["Very Negative", "Negative"]:
            return "Negativo"
        return "Neutral"
    except:
        return "Neutral"

# Carga del archivo
with st.sidebar:
    uploaded_file = st.file_uploader("📂 Sube un archivo CSV con una columna llamada 'opinion'", type=["csv"])

if uploaded_file:
    try:
        try:
            df = pd.read_csv(uploaded_file, delimiter=';')
            if 'opinion' not in df.columns:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)

        if "opinion" not in df.columns:
            st.error("⚠️ El archivo debe contener una columna llamada 'opinion'.")
            st.stop()

        st.success("✅ Archivo cargado correctamente")
        df = df.dropna(subset=['opinion'])
        df = df.head(20)
        opinions = df["opinion"].astype(str).tolist()

        # Palabras frecuentes
        stop_words = list(stopwords.words("spanish"))
        vectorizer = CountVectorizer(stop_words=stop_words)
        X = vectorizer.fit_transform(opinions)
        words = vectorizer.get_feature_names_out()
        word_sums = np.array(X.sum(axis=0)).flatten()
        word_freq = dict(zip(words, word_sums))
        top_indices = np.argsort(word_sums)[::-1][:10]
        top_words = [words[i] for i in top_indices]
        top_freq = [word_sums[i] for i in top_indices]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ☁️ Nube de Palabras")
            fig_wc, ax_wc = plt.subplots(figsize=(8, 4), facecolor='white')
            wordcloud = WordCloud(background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

        with col2:
            st.markdown("### 🔟 Palabras Más Frecuentes")
            fig_bar, ax_bar = plt.subplots(figsize=(8, 4), facecolor='white')
            bar_colors = ['#4B8BBE', '#306998', '#FFE873', '#FFD43B', '#646464', '#9C9C9C', '#3E8E7E', '#77C1B3', '#F9A03F', '#D74B4B']
            bars = ax_bar.bar(top_words, top_freq, color=bar_colors[:len(top_words)], edgecolor='black')
            for bar, count in zip(bars, top_freq):
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(int(count)),
                            va='bottom', ha='center', fontsize=12)
            ax_bar.set_ylabel("Frecuencia")
            plt.xticks(rotation=30, ha='right')
            st.pyplot(fig_bar)

        # Clasificación de sentimiento
        st.markdown("### 📊 Clasificación de Sentimiento")
        sentiments = [get_sentiment(op) for op in opinions]
        df_result = pd.DataFrame({"opinion": opinions, "sentimiento": sentiments})
        st.dataframe(df_result)

        st.markdown("### 📈 Distribución de Sentimientos")
        dist = df_result["sentimiento"].value_counts()
        pie_colors = ['#38ada9', '#f6b93b', '#e55039']
        fig_sent, ax_sent = plt.subplots(figsize=(6,4), facecolor='white')
        wedges, texts, autotexts = ax_sent.pie(
            dist,
            labels=dist.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=pie_colors[:len(dist)],
            textprops={'fontsize': 12}
        )
        ax_sent.axis('equal')
        st.pyplot(fig_sent)

        # Interacción con los comentarios cargados
        st.markdown("### 🔍 Explora los Comentarios")
        selected_option = st.radio("¿Qué deseas conocer sobre los 20 comentarios?", ["Resumen general", "Temas discutidos"])

        full_text = " ".join(opinions)
        parser = PlaintextParser.from_string(full_text, Tokenizer("spanish"))
        summarizer = LsaSummarizer()

        if selected_option == "Resumen general":
            summary_sentences = summarizer(parser.document, 5)
            summary = " ".join(str(sentence) for sentence in summary_sentences)
            st.info(summary if summary else full_text)
        else:
            words_sorted = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            topic_words = [w for w, _ in words_sorted[:10]]
            st.success("**Temas más discutidos:** " + ", ".join(topic_words))

    except Exception as e:
        st.error(f"❌ Error leyendo el archivo o procesando: {e}")
else:
    st.info("📥 Por favor, sube un archivo CSV con una columna llamada 'opinion' desde la barra lateral.")
