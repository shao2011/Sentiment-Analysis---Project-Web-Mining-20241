#!/usr/bin/env python3

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from emoji_data import load
from features import doc_to_ngrams, preprocess
from tqdm import tqdm
import numpy as np
# from sentence_transformers import SentenceTransformer
import torch

from argparse import ArgumentParser
from cmdline import add_args
ap = ArgumentParser()
add_args(ap, ('general', 'preproc', 'linear', 'test'))
opt = ap.parse_args()

if opt.class_weight:
    opt.class_weight = "balanced"
else:
    opt.class_weight = None

from logging import debug, info, basicConfig
basicConfig(level=opt.log_level,
                    format='%(asctime)s %(message)s')

data_trn = load(opt.input_prefix)
data_tst = load(opt.test_prefix)

docs_trn, v, _ = doc_to_ngrams(data_trn.docs, min_df=opt.min_df,
                          cache_dir = opt.cache_dir,
                          dim_reduce = opt.dim_reduce,
                          c_ngmin = opt.c_ngmin,
                          c_ngmax = opt.c_ngmax,
                          w_ngmin = opt.w_ngmin,
                          w_ngmax = opt.w_ngmax,
                          lowercase = opt.lowercase, cache=False, use_cached=False)
docs_tst = preprocess(data_tst.docs,
    c_ngmin=opt.c_ngmin, c_ngmax=opt.c_ngmax,
    w_ngmin=opt.w_ngmin, w_ngmax=opt.w_ngmax,
    lowercase=opt.lowercase)

docs_tst = v.transform(docs_tst)

# batch_size = 64
# model = SentenceTransformer('dunzhang/stella_en_400M_v5', device='cuda' if torch.cuda.is_available() else 'cpu', trust_remote_code=True)
# doc_embeddings = []
# for i in tqdm(range(0, len(docs_tst), batch_size), desc="Generating embeddings"):
#     batch_docs = docs_tst[i:i+batch_size]
#     embeddings = model.encode(batch_docs, show_progress_bar=False, convert_to_numpy=True)
#     doc_embeddings.append(embeddings)
# docs_tst = np.vstack(doc_embeddings)

if opt.classifier == 'lr': 
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(dual=True, C=opt.C, verbose=0,
            class_weight=opt.class_weight)
else:
    # from sklearn.linear_model import SGDClassifier
    # m = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-4)
    # from sklearn.svm import LinearSVC
    # m = LinearSVC(dual=True, C=opt.C, verbose=0,
            # class_weight=opt.class_weight)
    from sklearn.svm import SVC
    m = SVC(kernel='rbf', gamma='scale')
    
if opt.mult_class == 'ovo':
    mc = OneVsOneClassifier
else:
    mc = OneVsRestClassifier
m = mc(m, n_jobs=opt.n_jobs)

m.fit(docs_trn, data_trn.labels)

pred = m.predict(docs_tst)

# for lab in pred:
#     print(lab)

with open(opt.test_prefix + '.labels', 'w') as fp:
    for lab in pred:
        fp.write(lab + '\n')
