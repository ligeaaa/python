from JPSTH.utils.read_data import read_from_txt


def get_sum_psth(psth):
    psth_sum = [0 for _ in range(len(psth[0]))]
    for i in range(len(psth)):
        for j in range(len(psth[i])):
            psth_sum[j] += psth[i][j]
    return psth_sum


def calculate_psth(reference_data: list,
                   target_data: list,
                   x_min: float = -0.2,
                   x_max: float = 0.2,
                   bin_size: float = 0.01):
    flag = 0
    psth = [[0 for _ in range(int((x_max - x_min)/bin_size))] for _ in range(len(reference_data))]

    for i, event_time in enumerate(reference_data):
        for j in range(flag, len(target_data)):
            if x_min <= target_data[j] - event_time <= x_max:
                flag = j
                while x_min <= target_data[j] - event_time <= x_max:
                    psth[i][int((target_data[j] - event_time - x_min) / bin_size)] += 1
                    j = j + 1
                break
    return psth


if __name__ == '__main__':
    select_name = 'datas/Neuron04a.txt'
    bottom_name = 'datas/Neuron04a.txt'
    reference_name = 'datas/Event04.txt'

    select_data = read_from_txt(select_name)
    bottom_data = read_from_txt(bottom_name)
    reference_data = read_from_txt(reference_name)

    psth_bottom_data = calculate_psth(reference_data, bottom_data, x_min=-0.2, x_max=0.2, bin_size=0.05)
    # psth_select_data = calculate_psth(reference_data, select_data, x_min=-0.2, x_max=0.2, bin_size=0.05)

    sum = get_sum_psth(psth_bottom_data)

    print()