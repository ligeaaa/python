import numpy as np
from CV2.utils.read_data import read_from_txt
from CV2.utils.signal_generator import generate_possion_process
from CV2.utils.signal_generator import generate_gamma_process


def get_cv(isis):
    isis = np.asarray(isis)
    cv = (np.std(isis)) / np.mean(isis)
    return cv


if __name__ == '__main__':
    file_name = 'datas/Neuron04a.txt'
    datas = read_from_txt(file_name)

    # 完全平均的spike trains的cv值为0
    # datas = np.linspace(1,1000,1000)
    # datas[1] = 2.5

    # datas = [i for i in range(0, 100)]
    # datas.append(1000000000)

    # datas = generate_possion_process(10000, 1)

    # datas = generate_gamma_process(10, 0.1, 10000)
    isis = [datas[i + 1] - datas[i] for i in range(len(datas) - 1)]
    print(get_cv(isis))
