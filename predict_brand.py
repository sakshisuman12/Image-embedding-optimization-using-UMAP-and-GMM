import argparse
import pickle
import math
import numpy as np
import json
import yaml

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
import itertools
from sklearn import mixture


def norm_pdf_multivariate(x, mu, det, inv, norm_const):
    if det == 0:
        raise NameError("The covariance matrix can't be singular")
    x_mu = np.matrix(x - mu)
    result = np.exp(-0.5 * (x_mu @ inv @ x_mu.T))
    return norm_const * result


def get_soft_probabilities(data, weights, means, covariances, path):
    n_clusters = len(means)
    prob = np.zeros((1, n_clusters))
    matrix_calculations = {}
    with open(path) as f:
        matrix_calculations = json.loads(f.read())

    for k in range(n_clusters):
        prob[0, k] = weights[k] * norm_pdf_multivariate(np.array(data), np.array(means[k]),
                                                        matrix_calculations["determinants"][k],
                                                        matrix_calculations["inverses"][k],
                                                        matrix_calculations["norm_const"][k])
    return prob[0]


def run_gmm(X, **hp):
    model = mixture.GaussianMixture(**hp)
    model.fit(X)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="conf file")
    return parser.parse_args()


def main(conf):
    data = {}
    with open(conf["paths"]["unpacked_train_test_embeddings"]) as f:
        data = json.loads(f.read())

    with open(conf["paths"]["kmeans_object"], "rb") as f:
        km = pickle.load(f)
    path = conf["paths"]["save_gmm_matrices"]
    y_pred = []
    for d in data["X_test"]:
        g = km.predict([d])[0]
        with open(conf["paths"]["umap_object"].format(group=str(g)), "rb") as f:
            mapper = pickle.load(f)
        emb = mapper.transform([d])
        with open(conf["paths"]["gmm_weights"].format(group=str(g))) as f:
            models = json.load(f)
        means = list(itertools.chain.from_iterable(models["means"]))
        var = list(itertools.chain.from_iterable(models["covariances"]))
        weights = list(itertools.chain.from_iterable(models["weights"]))
        labels = list(itertools.chain.from_iterable(models["labels"]))
        group_path = path.format(group=str(g))
        o_threshold = conf["clustering"]["outlier_threshold"]
        test_p = get_soft_probabilities(emb, weights, means, var, group_path).tolist()
        max_idx = int(np.argmax(test_p))
        y_pred.append(labels[max_idx] if test_p[max_idx] >= o_threshold else "other")

    res = {}
    res["test_precision"] = precision_score(data["y_test"], y_pred, average='weighted')
    res["test_recall"] = recall_score(data["y_test"], y_pred, average='weighted')

    with open(conf["paths"]["gmm_eval_performance"], "w") as f:
        f.write(json.dumps(res))


if __name__ == "__main__":
    args = parse_args()
    conf = {}
    with open(args.conf) as f:
        conf = yaml.safe_load(f.read())
    main(conf)
