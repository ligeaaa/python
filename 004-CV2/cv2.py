import math
import numpy as np
from CV2.utils.read_data import read_from_txt
from CV2.utils.signal_generator import generate_possion_process
from CV2.utils.signal_generator import generate_gamma_process
import matplotlib.pyplot as plt


def get_cv2(spk_train: list,
            time_min: float = 0,
            time_max: float = 10000,
            max_pair_mean: float = 0.1,
            isi_pair_bin: float = 0.01,
            dot_size: float = 0.5):
    r"""
    Calculate the coefficient of variation

    Parameters
    ----------
    spk_train: list
        The spike times for one single unit in seconds.
    time_min: float
        Only the timestamps in time interval [time_min, time_max] will be selected for analysis
    time_max: float
        Only the timestamps in time interval [time_min, time_max] will be selected for analysis
    max_pair_mean: float
        cv2 graph shows ISI variability versus Mean of ISI Pair. max_pair_mean specifies maximum of x axis in cv2 graph
    isi_pair_bin: float
        Bin size for calculation of cv2
    dot_size: float
        The size of the dot in the graph

    Returns
    -------
    xxxxx

    References
    ----------
    .. [1] Holt, Gary R., William R. Softky, Christof Koch, and Rodney
           J. Douglas. “Comparison of discharge variability in vitro
           and in vivo in cat visual cortex neurons.” Journal of
           Neurophysiology 75, no. 5 (1996): 1806-1814.

       [2] https://www.neuroexplorer.com/docs/reference/analysis/types/trainstruct/CVTwo.html
    """

    result = {}
    # Calculate the ISIs(interspike intervals) of the spk_train
    isis = [spk_train[i + 1] - spk_train[i] for i in range(len(spk_train) - 1) if time_min <= spk_train[i] <= time_max]
    # Calculate the mean of two adjacent ISIs
    mean_of_isi = [(isis[i + 1] + isis[i]) / 2 for i in range(len(isis) - 1)]
    # Calculate the coefficient of variation of the ISIs
    # For spike i we compute the standard deviation of two adjacent ISIs, divide the result by their mean
    cv2 = [(2 * (abs(isis[i + 1] - isis[i])) / (isis[i + 1] + isis[i] + 1e-9))
           for i in range(len(isis) - 1)]
    cv2 = np.array(cv2)

    value_mean_cv2_bins = []
    index_mean_cv2_bins = []
    for i in np.arange(isi_pair_bin, max_pair_mean + isi_pair_bin, isi_pair_bin):
        # Collate every cv2‘s in every bin and calculate the mean of cv2 in every bin
        indexs = [index for index, value in enumerate(mean_of_isi) if i - isi_pair_bin <= value < i]
        value_mean_cv2_bins.append(cv2[indexs])
        index_mean_cv2_bins.append(i - isi_pair_bin / 2)

    # plt.boxplot(value_mean_cv2_bins, positions=index_mean_cv2_bins, showcaps=True,
    #             showmeans=False, widths=0.005, whis=1, showfliers=False, manage_ticks=False)
    plt.scatter(mean_of_isi, cv2, s=dot_size)
    plt.xlim(0, max_pair_mean)
    plt.ylim(0)
    plt.show()

    cv2_mean_of_bin = []
    for value_mean_cv2_bin in value_mean_cv2_bins:
        cv2_mean_of_bin.append(np.mean(value_mean_cv2_bin))

    result['timestamps'] = spk_train
    result['cv2'] = cv2
    result['mean_cv2'] = sum(cv2) / len(cv2)
    result['mean_of_isi_pair'] = mean_of_isi
    result['mean_of_isi_bin_middle'] = index_mean_cv2_bins
    result['cv2_mean_of_bin'] = cv2_mean_of_bin

    return result


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
    print(get_cv2(datas))
