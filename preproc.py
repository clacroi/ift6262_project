import numpy as np
from os import listdir
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def normalize_images(x, x_fn, meanStd_dict):
    n_images = x.shape[0]
    norm_x = np.zeros(x.shape)
    for i in range(n_images - 1):
        fn = x_fn[i]
        mean, std = meanStd_dict[fn]
        for k in range(3):
            norm_x[i, k, :, :] = (x[i, k, :, :] - mean[2 - k]) / std[2 - k]
    return norm_x


def reconstruct_images(norm_x, x_fn, meanStd_dict):
    n_images = norm_x.shape[0]
    x = np.zeros(norm_x.shape)
    for i in range(n_images - 1):
        fn = x_fn[i]
        mean, std = meanStd_dict[fn]
        for k in range(3):
            x[i, k, :, :] = (norm_x[i, k, :, :] * std[2 - k]) + mean[2 - k]
    return x.astype('uint8')


def construct_seq_embeddings(train_fn, val_fn, captions_dict, tfidf_args, svd_args):
    tr_corpus = []
    for fn in train_fn:
        im_captions = captions_dict[fn.split(".")[0]]
        tr_corpus.append(" ".join(im_captions))

    val_corpus = []
    for fn in val_fn:
        im_captions = captions_dict[fn.split(".")[0]]
        val_corpus.append(" ".join(im_captions))

    # Construct tfidf vectors for train and val captions
    print("TFIDF vectorizing...")
    vectorizer = TfidfVectorizer(**tfidf_args)
    vectorizer.fit(tr_corpus + val_corpus)
    tr_tfidf = vectorizer.transform(tr_corpus)
    val_tfidf = vectorizer.transform(val_corpus)

    # Fit SVD transformation on train captions and
    # predict vectors for train and val captions
    print("Computing SVD...")
    svd = TruncatedSVD(**svd_args)
    svd.fit(tr_tfidf)
    print(str(svd.explained_variance_ratio_.sum()) + "% variance explained by " + str(svd_args['n_components']) + "components")
    tr_embeddings = svd.transform(tr_tfidf)
    val_embeddings = svd.transform(val_tfidf)

    # Construct dicts with constructed embeddings
    tr_embeddings_dict = {}
    val_embeddings_dict = {}

    for k in range(len(train_fn)):
        tr_embeddings_dict[train_fn[k].split(".")[0]] = tr_embeddings[k]

    for k in range(len(val_fn)):
        val_embeddings_dict[val_fn[k].split(".")[0]] = val_embeddings[k]

    return tr_embeddings, val_embeddings


if __name__ == "__main__":
    
    print("Loading data...")
    # Compute captions embeddings
    with open("./Data/train_images_fn.pkl", 'rb') as input:
        train_fn = pickle.load(input)

    with open("./Data/val_images_fn.pkl", 'rb') as input:
        val_fn = pickle.load(input)

    with open("../Data/inpainting/dict_key_imgID_value_caps_train_and_valid.pkl", 'rb') as input:
        captions_dict = pickle.load(input, encoding='latin1')

    # Compute train and validation captions embeddings
    print("Computing caption embeddings...")
    tfidf_args = {'max_df':0.5, 'min_df':10e-5, 'smooth_idf':True}
    svd_args = {'n_components':2048, 'n_iter':7}
    tr_embeddings, val_embeddings = construct_seq_embeddings(train_fn, val_fn, captions_dict, tfidf_args, svd_args)

    # Save dicts containing train and validation embeddings
    print("Saving results...")
    with open('./Data/train_embeddings_v20', 'wb') as output:
        pickle.dump(tr_embeddings, output, pickle.HIGHEST_PROTOCOL)

    with open('./Data/val_embeddings_v20', 'wb') as output:
        pickle.dump(val_embeddings, output, pickle.HIGHEST_PROTOCOL)
