import pandas as pd
import re
import nltk
import tkinter as tk
from tkinter import ttk, scrolledtext, Frame, Label
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer
from utilis import preprocess, expand_keywords, preprocess_words, identificar_topicos
from collections import Counter
import random

nltk.download('stopwords')
nltk.download('wordnet')

# Pré-processamento
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Tópicos 
topics = {
    "location": ["location", "neighborhood", "area", "close", "near", "distance"],
    "staff": ["staff", "employee", "service", "reception", "friendly", "helpful"],
    "cleanliness": ["clean", "dirty", "spotless", "messy", "tidy"],
    "comfort": ["bed", "comfort", "noise", "quiet", "sleep", "room size"],
    "value": ["price", "expensive", "cheap", "worth", "deal", "value"],
    "amenities": ["breakfast", "pool", "gym", "spa", "wifi"],
    "checkin": ["check-in", "check-out", "late", "early", "delay"],
    "parking": ["parking", "garage", "valet", "lot"],
    "food": ["restaurant", "food", "meal", "dinner", "lunch", "buffet"]
}

topics_expanded = {k: expand_keywords(v) for k, v in topics.items()}
topics_stemmed = {k: preprocess_words(words) for k, words in topics_expanded.items()}

# Carregamento dos dados 
df = pd.read_csv("tripadvisor_hotel_reviews.csv")
df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)
df_sample['clean_review'] = df_sample['Review'].apply(preprocess)
df_sample['topics'] = df_sample['clean_review'].apply(lambda x: identificar_topicos(x, topics_stemmed))

# TF-IDF treinado no corpus completo, similaridade com reviews relevantes
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_sample['clean_review'])

# Interface gráfica
root = tk.Tk()
root.title("Analisador de Reviews por Tópico")
root.geometry("1000x700")
root.configure(bg="#e9ecef")

# Título
label_title = tk.Label(root, text="🔍 Analisador de Reviews por Tópico", font=("Segoe UI", 22, "bold"), bg="#e9ecef", fg="#2d3436")
label_title.pack(pady=(30, 10))

# Subtítulo
label_sub = tk.Label(root, text="Escolha um tópico para visualizar as avaliações mais relevantes", font=("Segoe UI", 14), bg="#e9ecef", fg="#636e72")
label_sub.pack(pady=(0, 20))

# Dropdown de tópicos
frame_topico = Frame(root, bg="#e9ecef")
frame_topico.pack(pady=10)
label_combo = tk.Label(frame_topico, text="Tópico:", font=("Segoe UI", 12, "bold"), bg="#e9ecef")
label_combo.pack(side=tk.LEFT, padx=(0, 8))
selected_topic = tk.StringVar()
combo = ttk.Combobox(frame_topico, textvariable=selected_topic, state="readonly", font=("Segoe UI", 12), width=20)
combo['values'] = [t.capitalize() for t in topics.keys()]
combo.pack(side=tk.LEFT)

# Botão 
def encontrar_reviews_similares_por_topico():
    chave = selected_topic.get().lower()
    if not chave or chave not in topics:
        output_text.config(state=tk.NORMAL)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "⚠️ Selecione um tópico válido.")
        output_text.config(state=tk.DISABLED)
        return

    # Filtra reviews com o tópico
    df_topico = df_sample[df_sample['topics'].apply(lambda ts: chave in ts)]
    if df_topico.empty:
        output_text.config(state=tk.NORMAL)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"❌ Nenhuma review com o tópico '{chave}'.")
        output_text.config(state=tk.DISABLED)
        return

    # Seleciona uma review aleatória com o tópico
    random_idx = random.choice(df_topico.index)
    review_base = df_sample.loc[random_idx, 'clean_review']
    review_original = df_sample.loc[random_idx, 'Review']
    rating_base = df_sample.loc[random_idx, 'Rating']
    tags_base = df_sample.loc[random_idx, 'topics']

    vec_base = vectorizer.transform([review_base])
    similarities = cosine_similarity(vec_base, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-11:][::-1]
    top_indices = [i for i in top_indices if i != random_idx][:10]

    # Exibir resultados 
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)

    # Cabeçalho
    output_text.insert(tk.END, f"📌 Tópico selecionado: {chave.capitalize()}\n\n", "section")

    # Review base - aleatória
    output_text.insert(tk.END, "🎯 Review de referência (aleatória):\n", "section")
    output_text.insert(tk.END, f"⭐ Nota: {rating_base}/5\n", "subinfo")
    output_text.insert(tk.END, f"🏷️ Tópicos: {' | '.join(tags_base)}\n", "subinfo")
    output_text.insert(tk.END, f"📝 {review_original}\n\n", "review")

    # Reviews semelhantes
    output_text.insert(tk.END, "📋 Top 10 Reviews mais semelhantes:\n\n", "section")

    for i, idx in enumerate(top_indices, start=1):
        review = df_sample.loc[idx, 'Review']
        rating = df_sample.loc[idx, 'Rating']
        tags = ' | '.join(df_sample.loc[idx, 'topics'])
        similarity = similarities[idx]

        output_text.insert(tk.END, f"{i}. 🔗 Similaridade: {similarity:.4f}   ⭐ Nota: {rating}/5   🏷️ Tópicos: {tags}\n", "subinfo")
        output_text.insert(tk.END, f"📝 {review}\n", "review")
        output_text.insert(tk.END, "──────────────────────────────────────────────────────────────────────────────\n", "subinfo")

    output_text.config(state=tk.DISABLED)

btn_topico_similares = tk.Button(
    root,
    text="🔍 Encontrar Reviews Similares por Tópico",
    font=("Segoe UI", 13, "bold"),
    bg="#6c5ce7",
    fg="white",
    activebackground="#a29bfe",
    activeforeground="white",
    command=encontrar_reviews_similares_por_topico,
    relief=tk.FLAT,
    bd=0,
    padx=20,
    pady=10
)
btn_topico_similares.pack(pady=20)

# Campo de saída 
output_text = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    width=110,
    height=24,
    font=("Segoe UI", 11),
    bg="#ffffff",
    fg="#2d3436",
    borderwidth=1,
    relief=tk.GROOVE
)
output_text.pack(padx=20, pady=10)

# Estilos de destaque no texto
output_text.tag_configure("header", font=("Segoe UI", 11, "bold"), foreground="#0984e3")
output_text.tag_configure("review", font=("Segoe UI", 11), foreground="#2d3436")
output_text.tag_configure("section", font=("Segoe UI", 12, "bold"), foreground="#6c5ce7")
output_text.tag_configure("subinfo", font=("Segoe UI", 10, "italic"), foreground="#636e72")
output_text.config(state=tk.DISABLED)

# Distribuição de tópicos
topico_counts = Counter([t for ts in df_sample['topics'] for t in ts])
label_stats = tk.Label(
    root,
    text=f"📊 Distribuição de tópicos (top 5): {dict(topico_counts.most_common(5))}",
    font=("Segoe UI", 10),
    bg="#e9ecef",
    fg="#636e72"
)
label_stats.pack(pady=(0, 10))

root.mainloop()
