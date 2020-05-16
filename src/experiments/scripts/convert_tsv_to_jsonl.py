import tqdm

if __name__ == '__main__':
    import argparse
    from os import listdir
    import os
    import json
    import re
    from os.path import isfile, join

    import numpy as np
    import json

    dataset_path = "../../data/bios/"


    for dset_name in ["dev", "test", "train"]:
        with open(f"{dataset_path}/{dset_name}.tsv") as f:
            with open(f"{dataset_path}/{dset_name}.jsonl", "w") as fo:
                headers = f.readline()

                for line in tqdm.tqdm(f):
                    line = line.strip()
                    if not line:
                        continue
                    line = line.split("\t")
                    ex = {"text": line[0], "label": int(line[1]) }

                    fo.write(json.dumps(ex))
                    fo.write("\n")



