import argparse
import pickle
import numpy as np
import json
import yaml

from umap import UMAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="conf file")
    return parser.parse_args()


def main(conf, X_train, X_test, y_train, y_test, groups={}):
    hp = conf["dr"]["umap"]
    if len(groups) == 0:
        with open(conf["paths"]["group_mappings"]) as f:
            groups = json.load(f)

    for g in groups:
        idxs = np.where(np.in1d(y_train, list(groups[str(g)].keys())))[0]
        X = X_train[np.ix_(idxs)]

        mapper = UMAP(**hp).fit(X)
        embeddings = {"embedding": mapper.embedding_.tolist(), "group":g}
        embeddings["labels"] = [y_train[i] for i in idxs]
        path = conf["paths"]["grouped_umap_path"].format(group=str(g))
        with open(path, "w") as f:
            json.dump(embeddings, f)
        with open(conf["paths"]["umap_object"].format(group=str(g)), "wb") as f:
            pickle.dump(mapper, f)


if __name__ == "__main__":
    args = parse_args()
    conf = yaml.safe_load(open(args.conf).read())
    with open(conf["paths"]["unpacked_train_test_embeddings"]) as f:
        data = json.loads(f.read())
    main(conf, np.array(data["X_train"]).astype(np.uint8),
         np.array(data["X_test"]).astype(np.uint8), data["y_train"], data["y_test"])