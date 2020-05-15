import pickle

def load_data(mode: str = "train"):

    with open("/home/nlp/ravfogs/inlp-final/nullspace_projection/data/biasbios/" + mode  + ".pickle", "rb") as f:
        data = pickle.load(f)
    return data

train = load_data("train")
dev = load_data("dev")
test = load_data("test")

print(train[0])