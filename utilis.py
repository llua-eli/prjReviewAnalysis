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

safe_nltk_download('stopwords')
safe_nltk_download('wordnet')
safe_nltk_download('punkt')
safe_nltk_download('averaged_perceptron_tagger')

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
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    processed_tokens = [
        stemmer.stem(lemmatizer.lemmatize(word.lower(), get_wordnet_pos(pos)))
        for word, pos in tagged_tokens if word.lower() not in stop_words
    ]
    return ' '.join(processed_tokens)

@lru_cache(maxsize=None)
def get_synonyms(word: str) -> list:
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().lower()
            if '_' not in name:
                syns.add(name)
    return list(syns)

def expand_keywords(words: list) -> list:
    expanded = set(words)
    for word in words:
        expanded.update(get_synonyms(word))
    return list(expanded)

def preprocess_words(word_list: list) -> list:
    return [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in word_list]

def identificar_topicos(texto: str, topics_stemmed: dict) -> list:
    if not texto.strip():
        return ['other']
    tokens = set(texto.split())
    return [topico for topico, palavras in topics_stemmed.items() if tokens.intersection(palavras)] or ['other']
