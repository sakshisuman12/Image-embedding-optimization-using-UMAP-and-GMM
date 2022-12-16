import argparse
import pickle

import numpy as np
import json
import yaml
from collections import defaultdict
from collections import Counter
from sklearn.cluster import KMeans


def save_groups(centroids_map, y_train, path):
    actual_labels = {}
    for i, c in enumerate(centroids_map):
        actual_labels[str(c)] = [y_train[j] for j in centroids_map[c]]
        actual_labels[str(c)] = dict(Counter(actual_labels[str(c)]))
    with open(path, "w") as f:
        json.dump(actual_labels, f)


def run_kmeans(X_train, **hp):
    km = KMeans(**hp)
    km.fit(X_train)
    return km


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="conf file")
    return parser.parse_args()


def main(conf, X_train, X_test, y_train, y_test):
    n_clusters = len(set(y_train))
    kmeans_hp = conf["clustering"]["kmeans"]
    if kmeans_hp["n_clusters"] == "auto":
        kmeans_hp["n_clusters"] = n_clusters
    results = {}
    km = run_kmeans(X_train, **kmeans_hp)
    results["km_labels"] = km.labels_.tolist()
    results["centroids"] = km.cluster_centers_.tolist()
    with open(conf["paths"]["kmeans_object"], "wb") as f:
        pickle.dump(km, f)
    with open(conf["paths"]["kmeans_centroids"], "w") as f:
        json.dump(results, f)
    centroids_map = {i: np.where(km.labels_ == i)[0] \
                     for i in km.labels_}
    cluster_group_path = conf["paths"]["group_mappings"]
    save_groups(centroids_map, y_train, cluster_group_path)




if __name__ == "__main__":
    args = parse_args()
    conf = yaml.safe_load(open(args.conf).read())
    with open(conf["paths"]["unpacked_train_test_embeddings"]) as f:
        data = json.loads(f.read())
    main(conf, np.array(data["X_train"]).astype(np.uint8),
         np.array(data["X_test"]).astype(np.uint8), data["y_train"], data["y_test"])
