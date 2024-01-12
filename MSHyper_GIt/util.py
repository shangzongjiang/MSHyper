from functools import reduce
import os
import random
import sys

import numpy
import numpy as np
import seaborn as sn
import torch
from matplotlib import pyplot as plt
from thop import profile, clever_format
from torch import nn
from ptflops import get_model_complexity_info
from sklearn import utils as skutils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
GLOBAL_SEED = 2022


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def count_parameters(model, inputs_shape):
    macs, params = get_model_complexity_info(model, tuple(inputs_shape[1:]), as_strings=True,
										print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    inputs = torch.rand(*inputs_shape).cuda()
    flops, params = profile(model, inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)


def matrixLoader2(adj_file):
    print(adj_file)
    adj = np.loadtxt(adj_file)
    return adj

def matrixLoader(adj_file, node_num=15):
    print(adj_file)
    adj = np.zeros([node_num, node_num], dtype=np.float)
    row = 0
    with open(adj_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip('\n').split('\t')
            adj[row:] = data[0:node_num]
            row += 1
    return adj

def set_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def _init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def data_loader(path):
    train_data = np.load(path)
    train_x, train_y = train_data['data'], train_data['target']
    return train_x


def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)


def makedir(path):
    os.makedirs(path, exist_ok=True)
    if not os.path.exists:
        print(f"[+] Created directory in {path}")


def paint(text, color="green"):
    """
    :param text: string to be formatted
    :param color: color used for formatting the string
    :return:
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    if color == "blue":
        return OKBLUE + text + ENDC
    elif color == "green":
        return OKGREEN + text + ENDC


def plot_pie(target, prefix, path_save, class_map=None, verbose=False):
    """
    Generate a pie chart of activity class distributions
    :param target: a list of activity labels corresponding to activity data segments
    :param prefix: data split, can be train, val or test
    :param path_save: path for saving the activity distribution pie chart
    :param class_map: a list of activity class names
    :param verbose:
    :return:
    """

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    if not class_map:
        class_map = [str(idx) for idx in range(len(set(target)))]

    color_map = sn.color_palette(
        "husl", n_colors=len(class_map)
    )  # a list of RGB tuples

    target_dict = {
        label: np.sum(target == label_idx) for label_idx, label in enumerate(class_map)
    }
    target_count = list(target_dict.values())
    if verbose:
        print(f"[-] {prefix} target distribution: {target_dict}")
        print("--" * 50)

    fig, ax = plt.subplots()
    ax.axis("equal")
    explode = tuple(np.ones(len(class_map)) * 0.05)
    patches, texts, autotexts = ax.pie(
        target_count,
        explode=explode,
        labels=class_map,
        autopct="%1.1f%%",
        shadow=False,
        startangle=0,
        colors=color_map,
        wedgeprops={"linewidth": 1, "edgecolor": "k"},
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.set_title(dataset)
    ax.legend(loc="center left", bbox_to_anchor=(1.2, 0.5))
    plt.tight_layout()
    # plt.show()
    save_name = os.path.join(path_save, prefix + ".png")
    fig.savefig(save_name, bbox_inches="tight")
    plt.close()


def plot_segment(
        data, target, index, prefix, path_save, num_class, target_pred=None, class_map=None
):
    """
    Plot a data segment with corresonding activity label
    :param data: data segment
    :param target: ground-truth activity label corresponding to data segment
    :param index: index of segment in dataset
    :param prefix: data split, can be train, val or test
    :param path_save: path for saving the generated plot
    :param num_class: number of activity classes
    :param target_pred: predicted activity label corresponding to data segment
    :param class_map: a list of activity class names
    :return:
    """

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    if not class_map:
        class_map = [str(idx) for idx in range(num_class)]

    gt = int(target)
    title_color = "black"

    if target_pred is not None:
        pred = int(target_pred)
        msg = f"#{int(index)}     ground-truth:{class_map[gt]}     prediction:{class_map[pred]}"
        title_color = "green" if gt == pred else "red"
    else:
        msg = "#{int(index)}     ground-truth:{class_map[gt]}            "

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(data.numpy())
    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(-5, 5)
    ax.set_title(msg, color=title_color)
    plt.tight_layout()
    save_name = os.path.join(
        path_save,
        prefix + "_" + class_map[int(target)] + "_" + str(int(index)) + ".png",
    )
    fig.savefig(save_name, bbox_inches="tight")
    plt.close()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name, fmt=":4f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


def sliding_window(x, y, window, stride, scheme="last"):
    data, target = [], []
    start = 0
    while start + window < x.shape[0]:
        end = start + window
        x_segment = x[start:end]
        if scheme == "last":
            # last scheme: : last observed label in the window determines the segment annotation
            y_segment = y[start:end][-1]
        elif scheme == "max":
            # max scheme: most frequent label in the window determines the segment annotation
            y_segment = np.argmax(np.bincount(y[start:end]))
        data.append(x_segment)
        target.append(y_segment)
        start += stride

    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.int64)

    return data, target