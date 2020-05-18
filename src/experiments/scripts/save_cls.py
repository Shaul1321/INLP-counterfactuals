if __name__ == '__main__':
    import argparse
    from os import listdir
    import os
    from nltk.tree import Tree
    import json
    import re
    from os.path import isfile, join
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', action='store')
    parser.add_argument('-o', '--output-path', action='store')
    # parser.add_argument('-n', '--label-name', action='store')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    instance = None
    label = None
    cls_vectors = []
    really_last_vectors = []
    with open(args.input_path, 'r') as f:
            # with open(args.output_path, 'w') as fo:
            for line in f:
                line = line.strip()
                if not line:
                    assert instance is not None and label is not None
                    # instance[args.label_name] = label

                    # sentence = instance['text']
                    # # parsed_text = Tree.fromstring(sentence)
                    # # parsed_text.set_label(int(label))
                    #
                    # # new_text = re.sub(r' {2,}', ' ', str(parsed_text).replace('\n', ''))
                    # # print(instance)
                    # # print(label)
                    # # # mask text here
                    # # exit()
                    # new_instance = {'text': sentence, 'label': instance['label'],
                    #                 'highlight': label['highlight'], 'highlight_text': label['highlight_text'],
                    #                 'highlight_length': label['highlight_length']}
                    #
                    # fo.write(json.dumps(new_instance))
                    # fo.write('\n')

                    cls_vectors.append(label['hidden_layer12_cls'])
                    really_last_vectors.append(label['really-last_cls'])

                    instance = None
                    label = None

                if line.startswith('input'):
                    line = line[line.index(':')+1:].strip()
                    instance = json.loads(line)
                elif line.startswith('prediction'):
                    line = line[line.index(':')+1:].strip()
                    label = json.loads(line)

    split_name = os.path.basename(args.input_path).split('.')[0]
    np.save(args.output_path + f"/{split_name}_cls", np.array(cls_vectors))
    np.save(args.output_path + f"/{split_name}_really-last_cls", np.array(really_last_vectors))