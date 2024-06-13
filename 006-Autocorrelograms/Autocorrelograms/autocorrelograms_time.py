from Autocorrelograms.utils.read_data import read_from_txt
from autocorrelograms import Autocorrelograms
import matplotlib.pyplot as plt


class AutocorrelogramsTime:
    def __init__(self,
                 select_data: list,
                 x_min: float = -0.2,
                 x_max: float = 0.2,
                 bin_size: float = 0.005,
                 start: float = 0,
                 duration: float = 10,
                 shift: float = 1,
                 number_of_shift: int = 20):
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
        start: float
            Start of the first sliding window in seconds.
        duration: float
            Duration of the sliding window in seconds.
        shift: float
            How much sliding window is shifted each time.
        number_of_shift: int
            The number of sliding windows to be used.


        References
        ----------
        .. [1] https://www.neuroexplorer.com/docs/reference/analysis/types/trainstruct/AutoCorrVersusTime.html
        """
        self.select_data = select_data
        self.x_min = x_min
        self.x_max = x_max
        self.bin_size = bin_size
        self.start = start
        self.duration = duration
        self.shift = shift
        self.number_of_shift = number_of_shift
        self.autocorrelograms_time = self._get_autocorrelograms_time()

    def _get_autocorrelograms_time(self):
        r"""
        Calculate the autocorrelograms of every bin.

        Returns
        -------
        This function return the list that include the autocorrelograms of every bin.
        Assume result = [[n_1,...,n_i], ..., [n_1,...,n_i]], where n means the count of
        spike in this bin.
        """
        result = []
        for i in range(self.number_of_shift):
            # Start_time and end_time increase depend on self.shift
            start_time = self.start + i * self.shift
            end_time = self.start + i * self.shift + self.duration
            # Calculate the autocorrelogram in this bin depends on the start_time and end_time
            autocorrelogram = Autocorrelograms(self.select_data, whether_select=True, select_data_from=start_time,
                                               select_data_to=end_time).bin_count
            result.append(autocorrelogram)

        return result

    def draw_autocorrelograms(self):
        r"""
        Draw the graph shows the 006-Autocorrelograms Versus Time
        """
        z = [list(row) for row in zip(*self.autocorrelograms_time)]
        x = [start_time for start_time in range(self.number_of_shift)]
        y = [self.x_min + i * self.bin_size for i in range(int((self.x_max - self.x_min) / self.bin_size))]
        plt.pcolormesh(x, y, z)
        plt.colorbar()
        plt.show()



if __name__ == '__main__':
    file_name = 'datas/Neuron04a.txt'
    datas = read_from_txt(file_name)
    test = AutocorrelogramsTime(datas)
    test.draw_autocorrelograms()
    print()
