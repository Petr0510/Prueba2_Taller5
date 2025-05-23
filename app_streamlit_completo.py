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

nltk.download("punkt")
nltk.download("stopwords")

st.set_page_config(
    page_title="Customer Feedback Analysis",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("📝 Customer Feedback Analysis")

@st.cache_resource(show_spinner="Cargando modelo de sentimiento...")
def get_sentiment_pipeline():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

sentiment_pipe = get_sentiment_pipeline()

def get_sentiment(text):
    try:
        result = sentiment_pipe(text)
        if result and len(result) > 0:
            label = result[0]['label']
            if label in ["Very Positive", "Positive"]:
                return "Positivo"
            elif label == "Neutral":
                return "Neutral"
            elif label in ["Very Negative", "Negative"]:
                return "Negativo"
            else:
                return "Neutral"
        else:
            return "Neutral"
    except Exception:
        return "Neutral"

uploaded_file = st.file_uploader("Sube tu archivo CSV con una columna llamada 'opinion'", type=["csv"])
if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, delimiter=';')
            if 'opinion' not in df.columns:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        if "opinion" not in df.columns:
            st.error("⚠️ El archivo debe contener una columna llamada 'opinion'.")
            st.stop()
        else:
            st.success("Archivo cargado correctamente.")
            st.write("Vista previa de las opiniones:")
            st.dataframe(df.head())

            opinions = df["opinion"].dropna().astype(str).tolist()
            stop_words = list(stopwords.words("spanish"))
            vectorizer = CountVectorizer(stop_words=stop_words)
            X = vectorizer.fit_transform(opinions)
            words = vectorizer.get_feature_names_out()
            word_sums = np.array(X.sum(axis=0)).flatten()
            top_indices = np.argsort(word_sums)[::-1][:10]
            top_words = [words[i] for i in top_indices]
            top_freq = [word_sums[i] for i in top_indices]

            # --- Nube de palabras ---
            st.subheader("☁️ Nube de Palabras")
            word_freq = dict(zip(words, word_sums))
            fig_wc, ax_wc = plt.subplots(figsize=(8, 4), facecolor='white')
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_freq)
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            fig_wc.patch.set_facecolor('white')
            st.pyplot(fig_wc)
            plt.close(fig_wc)

            # --- Top 10 palabras más frecuentes ---
            st.subheader("🔟 Palabras Más Frecuentes")
            fig_bar, ax_bar = plt.subplots(figsize=(9, 5), facecolor='white')
            bar_colors = ['#6a89cc', '#38ada9', '#b8e994', '#f6b93b', '#e55039',
                          '#4a69bd', '#60a3bc', '#78e08f', '#fa983a', '#eb2f06']
            bars = ax_bar.bar(top_words, top_freq, color=bar_colors[:len(top_words)], edgecolor='black')
            ax_bar.set_ylabel("Frecuencia", fontsize=14, weight='bold')
            ax_bar.set_title("Top 10 Palabras Más Frecuentes", fontsize=18, weight='bold', pad=15)
            ax_bar.set_facecolor('white')
            fig_bar.patch.set_facecolor('white')
            for bar, count in zip(bars, top_freq):
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(int(count)), va='bottom', ha='center', color='#222831', fontsize=15, fontweight='bold')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

            # --- Clasificación de sentimiento ---
            st.subheader("📊 Clasificación de Sentimiento")
            sentiments = [get_sentiment(op) for op in opinions]
            df_result = pd.DataFrame({
                "opinion": opinions,
                "sentiment": sentiments
            })
            st.dataframe(df_result)

            # --- Pie chart ---
            st.subheader("📈 Distribución de Sentimientos")
            dist = df_result["sentiment"].value_counts()
            pie_colors = ['#38ada9', '#f6b93b', '#e55039']
            fig_sent, ax_sent = plt.subplots(figsize=(6,4), facecolor='white')
            wedges, texts, autotexts = ax_sent.pie(
                dist,
                labels=dist.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=pie_colors[:len(dist)],
                textprops={'color':"black", 'fontsize':14}
            )
            ax_sent.axis('equal')
            ax_sent.set_title("Distribución de Sentimientos", fontsize=15, weight='bold')
            fig_sent.patch.set_facecolor('white')
            plt.setp(autotexts, size=14, weight="bold")
            plt.setp(texts, size=14)
            plt.tight_layout()
            st.pyplot(fig_sent)
            plt.close(fig_sent)

            # --- Interacción con los primeros 20 comentarios ---
            st.subheader("🗂 Interacción con los primeros 20 comentarios")
            top_20 = df_result.head(20)
            st.write("Primeros 20 comentarios:")
            st.dataframe(top_20)

            action = st.radio(
                "¿Qué deseas hacer con estos comentarios?",
                ("Mostrar resumen", "Mostrar temas más discutidos")
            )

            if action == "Mostrar resumen":
                text_20 = " ".join(top_20["opinion"])
                parser_20 = PlaintextParser.from_string(text_20, Tokenizer("spanish"))
                summarizer = LsaSummarizer()
                summary_20 = summarizer(parser_20.document, 5)
                resumen = " ".join(str(sentence) for sentence in summary_20)
                st.info(resumen if resumen else text_20)
            else:
                st.subheader("🧵 Temas más discutidos")
                vectorizer_20 = CountVectorizer(stop_words=stop_words)
                X20 = vectorizer_20.fit_transform(top_20["opinion"])
                words_20 = vectorizer_20.get_feature_names_out()
                word_sums_20 = np.array(X20.sum(axis=0)).flatten()
                top_idx_20 = np.argsort(word_sums_20)[::-1][:10]
                temas = [words_20[i] for i in top_idx_20]
                frecuencias = [word_sums_20[i] for i in top_idx_20]
                st.table(pd.DataFrame({"Tema": temas, "Frecuencia": frecuencias}))

    except Exception as e:
        st.error(f"❌ Error leyendo el archivo o procesando: {e}")
else:
    st.info("Por favor, sube un archivo CSV con una columna llamada 'opinion'.")