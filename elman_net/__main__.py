from typing import List, Tuple


def prepare_text(text: str) -> List[Tuple[str, str]]:
    return [(text[i], text[i + 1]) for i in range(len(text) - 1)]


def train_test_split(data, train_w, test_w):
    total = train_w + test_w
    train_f = train_w / total
    train_n = round(len(data) * train_f)
    return (data[0:train_n], data[train_n:])


# def train_elman_net():
#     from elman_net.elman_net import ElmanNetwork

#     with open("./dataset/dirtydata.txt") as f:
#         text = f.read()

#     data = prepare_text(text)

#     train_data, test_data = train_test_split(data, 80, 20)

#     net = ElmanNetwork([163, 200, 163])
#     net.gradient_descent(train_data, 1, 100, 0.05, test_data)


def train_test_net():
    from elman_net.test_net import Network
    import numpy as np
    import random
    import math

    # load the dataset
    dataset = []
    with open("test_dataset/diagnosis.data") as dataset_f:
        for line in dataset_f:
            items = line.strip().split("\t")

            temp = float(items[0])
            rest = list(map(lambda x: 1 if x == "yes" else 0, items[1:]))

            x = [temp]
            x.extend(rest[:-2])
            y = rest[-2:]

            dataset.append((x, y))

    # split dataset into training and testing data
    random.shuffle(dataset)
    split_i = math.ceil(len(dataset) * 0.8)
    train_data = dataset[:split_i]
    test_data = dataset[split_i:]

    # create ANN
    net = Network([6, 4, 2])

    # start learning
    net.gradient_descent(train_data, 100, 10, 0.05, test_data=test_data)

    # with open("test.txt", "w") as f:
    #     for i in inputs:
    #         print(f"Input: {i}, output: {net.feedforward(i)}", file=f)


if __name__ == "__main__":
    # train_elman_net()
    train_test_net()
