import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from functools import lru_cache

def safe_nltk_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

# Garante que os recursos do NLTK estão disponíveis
safe_nltk_download('stopwords')
safe_nltk_download('wordnet')
safe_nltk_download('punkt')
safe_nltk_download('averaged_perceptron_tagger')
safe_nltk_download('punkt_tab')
safe_nltk_download('averaged_perceptron_tagger_eng')

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Mapeia POS tag do nltk para o formato do wordnet lemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text: str) -> str:
    """Limpa, tokeniza e lematiza o texto."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    processed_tokens = [
        lemmatizer.lemmatize(word.lower(), get_wordnet_pos(pos))
        # Para usar stemming junto, descomente a linha abaixo e comente a de cima
        # stemmer.stem(lemmatizer.lemmatize(word.lower(), get_wordnet_pos(pos)))
        for word, pos in tagged_tokens if word.lower() not in stop_words
    ]
    return ' '.join(processed_tokens)

@lru_cache(maxsize=None)
def get_synonyms(word: str) -> list:
    """Retorna lista de sinônimos para uma palavra usando WordNet."""
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().lower()
            if '_' not in name:
                syns.add(name)
    return list(syns)

def expand_keywords(words: list) -> list:
    """Expande lista de palavras com sinônimos."""
    expanded = set(words)
    for word in words:
        expanded.update(get_synonyms(word))
    return list(expanded)

def preprocess_words(word_list: list) -> list:
    """Lematiza (ou faz stemming) em lista de palavras."""
    return [lemmatizer.lemmatize(word.lower()) for word in word_list]
    # Para usar stemming junto, troque por:
    # return [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in word_list]

def identificar_topicos(texto: str, topics_stemmed: dict) -> list:
    """Retorna lista de tópicos presentes no texto."""
    if not texto.strip():
        return ['other']
    tokens = set(texto.split())
    return [topico for topico, palavras in topics_stemmed.items() if tokens.intersection(palavras)] or ['other']
