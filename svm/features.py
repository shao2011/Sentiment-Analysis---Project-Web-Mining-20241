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

def identity(x):
    return x

def preprocess(docs, c_ngmin=1, c_ngmax=1,
        w_ngmin=1, w_ngmax=1, lowercase=None):0
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
                  cache_dir='.cache', **kwargs):
    """ Return bag-of-n-grams features for the given document set with progress tracking """
    param = {
        'c_ngmax': 1, 'c_ngmin': 1, 'w_ngmax': 1, 'w_ngmin': 1,
        'min_df': 1,
        'sublinear': True,
        'norm': 'l2',
        'max_features': None,
        'input_name': None,
        'lowercase': None,
        'dim_reduce': None
    }
    for k, v in kwargs.items(): 
        param[k] = v

    # Cache filename generation
    if param['input_name'] and use_cached or cache:
        os.makedirs(cache_dir, exist_ok=True)
        paramstr = ','.join([k + '=' + str(param[k]) for k in sorted(param)])
        cachefn = 'vectorizer-' + hashlib.sha224(paramstr.encode('utf-8')).hexdigest() + '.z'
        cachefn = os.path.join(cache_dir, cachefn)

    # Use cached vectorizer if available
    if use_cached and os.path.exists(cachefn):
        info('Using cached vectorizer: {}'.format(cachefn))
        with open(cachefn, 'r') as fp:
            v = joblib.load(cachefn)
            vectors = joblib.load(cachefn.replace('vectorizer-', 'vectors-'))
    else:
        # Preprocessing step with tqdm for progress tracking
        info("Preprocessing documents...")
        features = list(tqdm(preprocess(docs, 
                                        c_ngmin=param['c_ngmin'],
                                        c_ngmax=param['c_ngmax'], 
                                        w_ngmin=param['w_ngmin'], 
                                        w_ngmax=param['w_ngmax'], 
                                        lowercase=param['lowercase']),
                             desc="Preprocessing Docs"))

        # Vectorization step with progress tracking
        info("Vectorizing features...")
        v = TfidfVectorizer(analyzer=identity,
                            lowercase=(param['lowercase'] == 'all'),
                            sublinear_tf=param['sublinear'],
                            min_df=param['min_df'],
                            norm=param['norm'],
                            max_features=param['max_features'])
        vectors = v.fit_transform(tqdm(features, desc="Fitting TF-IDF Vectorizer"))

        # Save to cache if required
        if cache and param['input_name']:
            info('Saving vectorizer and vectors to cache...')
            joblib.dump(v, cachefn, compress=True)
            joblib.dump(vectors, cachefn.replace('vectorizer-', 'vectors-'), compress=True)

    # Dimensionality reduction step
    svd = None
    if param['dim_reduce']:
        info(f"Reducing dimensionality: {len(v.vocabulary_)} -> {param['dim_reduce']}...")
        svd = TruncatedSVD(n_components=param['dim_reduce'], n_iter=10)
        svd.fit(vectors)
        info(f"Explained variance: {svd.explained_variance_ratio_.sum()}")
        vectors = svd.transform(tqdm(vectors, desc="Applying Dimensionality Reduction"))

    return vectors, v, None


w_tokenizer = re.compile(r"\w+|[^ \t\n\r\f\v\w]+").findall