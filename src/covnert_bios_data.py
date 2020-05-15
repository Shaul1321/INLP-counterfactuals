import pickle
from typing import List

def load_data(mode: str = "train") -> List[dict]:

    with open("/home/nlp/ravfogs/inlp-final/nullspace_projection/data/biasbios/" + mode  + ".pickle", "rb") as f:
        data = pickle.load(f)
    return data

def process(data: List[dict], prof2ind, mode):

    sents, labels = [], []
    for d in data:
        sents.append(d["hard_text"])
        labels.append(prof2ind[d["p"]])

    with open(mode + ".tsv", "w", encoding = "utf-8") as f:
        f.write("sentence\tlabel\n")

        for s,l in zip(sents,labels):
            f.write(s + "\t" + l + "\n")


train = load_data("train")
dev = load_data("dev")
test = load_data("test")

professions = set([d["p"]] for d in train)
prof2ind = {p:i for i,p in enumerate(sorted(professions))}

with open("prof2ind.pickle", wb) as f:
    pickle.dump(prof2ind, f)


process(train, prof2ind, "train")
process(dev, prof2ind, "dev")
process(test, prof2ind, "test")