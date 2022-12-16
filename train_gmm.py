import math
import argparse
import numpy as np
import json
import yaml
from scipy.stats import multivariate_normal
from sklearn import mixture


def run_gmm(X, **hp):
    model = mixture.GaussianMixture(**hp)
    model.fit(X)
    return model


def save_det_inverse(means, covariances):
    n_clusters = len(means)
    norm_const = []
    determinants = []
    inverses = []
    size = len(means[0])
    for k in range(n_clusters):
        d = np.linalg.det(covariances[k])
        determinants.append(d.tolist())
        inverses.append(np.linalg.inv(covariances[k]).tolist())
        norm_const.append(1.0 / (math.pow((2 * np.pi), float(size) / 2) * math.pow(d, 1.0 / 2)))

    return determinants, inverses, norm_const


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="conf file")
    return parser.parse_args()


def main(conf):
    data = {}
    groups = {}
    with open(conf["paths"]["group_mappings"]) as f:
        groups = json.load(f)
    umap_paths = [conf["paths"]["grouped_umap_path"].format(group=str(g)) for g in groups]
    for path in umap_paths:
        with open(path) as f:
            data = json.load(f)
        result = {"means": [], "covariances": [], "labels": [], "weights": []}
        models = []
        determinants, inverses, norm_const = [], [], []
        performance = {}
        labels = set(data["labels"])
        for label in labels:
            X = np.array(data["embedding"])
            model = run_gmm(X, **conf["clustering"]["gmm"])
            result["means"].append(model.means_.tolist())
            result["covariances"].append(model.covariances_.tolist())
            result["weights"].append(model.weights_.tolist())
            result["labels"].append([label] * len(result["means"][-1]))
            det, inv, n_c = save_det_inverse(model.means_, model.covariances_)
            determinants.extend(det)
            inverses.extend(inv)
            norm_const.extend(n_c)
            models.append(model)
        matrix_path = conf["paths"]["save_gmm_matrices"].format(group=str(data["group"]))
        with open(matrix_path, "w") as f:
            json.dump(dict(determinants=determinants, inverses=inverses, norm_const=norm_const), f)
        with open(conf["paths"]["gmm_weights"].format(group=str(data["group"])), "w") as f:
            f.write(json.dumps(result))


if __name__ == "__main__":
    conf = {}
    args = parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f.read())
    main(conf)
