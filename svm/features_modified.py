import os.path
import re
import pickle, hashlib
import joblib
from logging import debug, info
from ngram import get_ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from tqdm import tqdm
import hashlib
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
import emoji
from nltk.stem import WordNetLemmatizer
from nltk import download
import torch
from sklearn.decomposition import TruncatedSVD

# Download required NLTK data
download('wordnet')
download('omw-1.4')

def identity(x):
    return x

def preprocess(docs, c_ngmin=1, c_ngmax=1,
        w_ngmin=1, w_ngmax=1, lowercase=None):
    # convert docs to word/char ngrams with optional case normaliztion
    # this would ideally be tha anlyzer parameter of the
    # vectorizer, but requires lambda - which breaks saving 
    features = []
    for doc in docs:
        # character n-grams
        if lowercase == 'char':
            docfeat = get_ngrams(doc.lower(),
                    ngmax=c_ngmax, ngmin=c_ngmin,
                    tokenizer=list)
        else:
            docfeat = get_ngrams(doc,
                    ngmax=c_ngmax, ngmin=c_ngmin,
                    tokenizer=list)
        # word n-grams
        if lowercase == 'word':
            docfeat.extend(get_ngrams(doc.lower(),
                        ngmax=w_ngmax, ngmin=w_ngmin,
                        append="W"))
        else:
            docfeat.extend(get_ngrams(doc,
                        ngmax=w_ngmax, ngmin=w_ngmin,
                        append="W"))
        features.append(docfeat)
    return features

def doc_to_ngrams(docs, use_cached=True, cache=True,
                 cache_dir='.cache', transformer_model='dunzhang/stella_en_400M_v5', **kwargs):
    """
    Return combined bag-of-n-grams and transformer embeddings features for the given document set with progress tracking.
    
    Returns:
        vectors (sparse matrix or ndarray): Combined feature matrix.
        v (TfidfVectorizer): Fitted TF-IDF vectorizer.
        None
    """
    # Define default parameters
    param = {
        'c_ngmax': 6, 'c_ngmin': 1, 'w_ngmax': 4, 'w_ngmin': 1,
        'min_df': 2,
        'sublinear': True,
        'norm': 'l2',
        'max_features': None,
        'input_name': 'emoji_data',
        'lowercase': 'all',
        'dim_reduce': None
    }
    # Update parameters with any additional keyword arguments
    for k, v in kwargs.items(): 
        param[k] = v

    # Generate a unique cache filename based on parameters and transformer model
    paramstr = ','.join([k + '=' + str(param[k]) for k in sorted(param)])
    transformer_cache_key = f"transformer_model={transformer_model}"
    cachefn = 'vectorizer-' + hashlib.sha224((paramstr + transformer_cache_key).encode('utf-8')).hexdigest() + '.z'
    cachefn = os.path.join(cache_dir, cachefn)

    # Check if cached vectorizer and vectors exist
    if use_cached and os.path.exists(cachefn):
        print(f'Using cached vectorizer and vectors: {cachefn}')
        with open(cachefn, 'rb') as fp:
            v = joblib.load(fp)
            vectors = joblib.load(cachefn.replace('vectorizer-', 'vectors-'))
    else:
        # Preprocessing step with tqdm for progress tracking
        print("[SUB-PROGRESS] Preprocessing documents...")
        features = list(tqdm(preprocess(docs, 
                                        c_ngmin=param['c_ngmin'],
                                        c_ngmax=param['c_ngmax'], 
                                        w_ngmin=param['w_ngmin'], 
                                        w_ngmax=param['w_ngmax'], 
                                        lowercase=param['lowercase']),
                             desc="Preprocessing Docs"))

        # Vectorization step with progress tracking
        print("[SUB-PROGRESS] Vectorizing TF-IDF features...")
        v = TfidfVectorizer(analyzer=identity,
                            lowercase=False,  # Already handled in preprocess
                            sublinear_tf=param['sublinear'],
                            min_df=param['min_df'],
                            norm=param['norm'],
                            max_features=param['max_features'])
        vectors_tfidf = v.fit_transform(tqdm(features, desc="Fitting TF-IDF Vectorizer"))

        # Generate transformer embeddings
        print("[SUB-PROGRESS] Generating transformer embeddings...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {device}")
        model = SentenceTransformer(transformer_model, device=device, trust_remote_code=True)
        # To handle large datasets, encode in batches
        batch_size = 64
        doc_embeddings = []
        for i in tqdm(range(0, len(docs), batch_size), desc="Generating embeddings"):
            batch_docs = docs[i:i+batch_size]
            embeddings = model.encode(batch_docs, show_progress_bar=False, convert_to_numpy=True)
            doc_embeddings.append(embeddings)
        doc_embeddings = np.vstack(doc_embeddings)
        # Convert embeddings to sparse matrix
        # doc_embeddings_sparse = csr_matrix(doc_embeddings)

        # Combine TF-IDF vectors with transformer embeddings
        # print("Combining TF-IDF and transformer embeddings...")
        # vectors = hstack([vectors_tfidf, doc_embeddings_sparse])
        # scaler = TruncatedSVD(n_components=1024)
        # vectors = scaler.fit_transform(doc_embeddings)
        print(f'[INFO] Dimensionality of features: {vectors.shape}')

        # Save to cache if required
        if cache and param['input_name']:
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Saving vectorizer and vectors to cache: {cachefn}')
            joblib.dump(v, cachefn, compress=True)
            joblib.dump(vectors, cachefn.replace('vectorizer-', 'vectors-'), compress=True)

    # Dimensionality reduction step (optional)
    if param['dim_reduce']:
        print(f"Reducing dimensionality: {vectors.shape[1]} -> {param['dim_reduce']}...")
        svd = TruncatedSVD(n_components=param['dim_reduce'], n_iter=10, random_state=42)
        vectors = svd.fit_transform(tqdm(vectors, desc="Applying Dimensionality Reduction"))
        print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2f}")

    return vectors, v, None


w_tokenizer = re.compile(r"\w+|[^ \t\n\r\f\v\w]+").findall


def doc_to_numseq(doc, vocab, tokenizer="char", pad=None):
    """ Transform given sequence of labels to numeric values 
    """
    from keras.preprocessing.sequence import pad_sequences
    oov_char = 1
    start_char = 2
    end_char = 3
    features = {k:v+4 for v,k in enumerate(vocab.keys())}
    X = []
    maxlen = 0
    for d in doc:
        x = [start_char]
        if tokenizer == "word":
            tokenizer = w_tokenizer
        elif tokenizer == "char":
            tokenizer = list
        tokens = tokenizer(d)
        for c in tokens:
            if c in features:
                x.append(features[c])
            else:
                x.append(oov_char)
        x.append(end_char)
        if len(x) > maxlen: maxlen = len(x)
        X.append(x)
    X = np.array(X)
    if pad:
        X = pad_sequences(X, maxlen=pad)
    return X, maxlen
