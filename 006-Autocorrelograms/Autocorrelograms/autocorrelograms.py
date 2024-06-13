from Autocorrelograms.utils.read_data import read_from_txt


class Autocorrelograms:
    def __init__(self,
                 select_data: list,
                 x_min: float = -0.2,
                 x_max: float = 0.2,
                 bin_size: float = 0.005,
                 whether_select: bool = False,
                 select_data_from: float = 0.0,
                 select_data_to: float = 1.0):
        r"""
        Calculate the coefficient of variation

        Parameters
        ----------
        select_data: list
            The data used to calculate autocorrelograms vs time
        x_min: float
            Event stimulus time point left boundary, in seconds.
        x_max: float
            Event stimulus time point right boundary, in seconds.
        bin_size: float
            Bin size in seconds
        whether_select: bool
            If false, then calculate the whole data's autocorrelograms
            If True, then calculate the data in range of [select_data_from, select_data_to]
        select_data_from: float
            Start of the time range in seconds.
        select_data_to: float
            End of the time range in seconds.

        References
        ----------
        .. [1] https://www.neuroexplorer.com/docs/reference/analysis/types/trainstruct/Autocorrelograms.html
        """
        self.whether_select = whether_select
        self.select_data_from = select_data_from
        self.select_data_to = select_data_to
        # preprocess the select data
        self.select_data = self._preprocess_data(select_data)
        self.x_min = x_min
        self.x_max = x_max
        self.bin_size = bin_size
        self.bin_count = self.get_autocorrelograms()

    def get_autocorrelograms(self):
        # calculate the count of bin
        bin_count = [0 for _ in range(int((self.x_max - self.x_min) / self.bin_size))]
        flag = 0
        for i in range(len(self.select_data)):
            for j in range(flag, len(self.select_data)):
                # don't calculate the distances from this spike to itself
                if j == i:
                    continue
                difference = self.select_data[j] - self.select_data[i]
                if difference < self.x_min:
                    flag = j
                if difference > self.x_max:
                    break
                if self.x_min <= difference < self.x_max:
                    while j < len(self.select_data) and self.x_min <= self.select_data[j] - self.select_data[i] < self.x_max:
                        if j != i:
                            # store the number of spike in the bin
                            bin_count[int((self.select_data[j] - self.select_data[i] - self.x_min) / self.bin_size)] += 1
                        j += 1
                    break
        return bin_count

    def draw_autocorrelograms(self):
        # Todo 画出自相关图
        pass

    def _preprocess_data(self, select_data):
        if self.whether_select:
            select_data = [data for data in select_data if self.select_data_from <= data <= self.select_data_to]
        return select_data


if __name__ == '__main__':
    file_name = 'datas/Neuron04a.txt'
    datas = read_from_txt(file_name)
    test = Autocorrelograms(datas, whether_select=True, select_data_from=0, select_data_to=10)
    test.draw_autocorrelograms()
    print()
