import math

from JPSTH.utils.read_data import read_from_txt


class JointPeriStimulusTimeHistogram:
    def __init__(self,
                 reference_data: list,
                 select_data: list,
                 bottom_data: list,
                 x_min: float = -0.2,
                 x_max: float = 0.2,
                 bin_size: float = 0.01,
                 normalization: str = 'Raw JPSTH',
                 matrix_scale: str = 'Color Scale'):
        r"""
        Calculate the Joint Peri-Stimulus Time Histogram

        Parameters
        ----------
        reference_data : list
            Specifies a reference neuron or event.
        select_data : list
            The neuron of event shown along the vertical axis (the vertical axis shows a neuron selected for analysis)
        bottom_data : list
            The neuron of event shown along the horizontal axis (the vertical axis shows a neuron selected for analysis)
        x_min : float
            Event stimulus time point left boundary, in seconds.
        x_max : float
            Event stimulus time point right boundary, in seconds.
        bin_size : float
            Bin size in seconds.
        normalization : str
            Scatter matrix normalization.
        matrix_scale : str
            An option on how to draw the scatter matrix (color or black and white).

        References
        ----------
        .. [1] M. H. J. Aertsen, G. L. Gerstein, M. K. Habib and G. Palm.
               Dynamics of Neuronal Firing Correlation: Modulation of
               “Effective Connectivity” J. Neurophysiol., Vol. 61, pp. 900-917, 1989.
           [2]Ito H, Tsuji S. Model dependence in quantification of spike
              interdependence by joint peri-stimulus time histogram[J].
              Neural computation, 2000, 12(1): 195-217.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.bin_size = bin_size
        # Calculate the number of bin
        self.bin_number = int((self.x_max - self.x_min) / self.bin_size)

        self.reference_data = reference_data
        self.select_data = select_data
        self.bottom_data = bottom_data
        # Todo 异步
        # Calculate the PSTH data of two neurons signals
        self.psth_select_data = self._get_psth(self.select_data)
        self.psth_bottom_data = self._get_psth(self.bottom_data)
        self.normalization = normalization
        self.matrix_scale = matrix_scale
        # Calculate the data of Joint PSTH
        self.original_jpsth = self._get_original_jpsth()

    def _get_original_jpsth(self):
        """
        Calculate $n_{ij}^{(k)}(u,v) = n_{i}^{(k)}(u) * n_{i}^{(k)}(u)$
        i, j: Spike trains of neuron i and neuron j.
        k   : The k-th record (corresponding to the time point in reference_data).
        u, v: Data at bin coordinates (u, v).

        The value in the (u, v) window of the JPSTH between neuron i and neuron j in the k-th trial.

        Returns
        -------
        jpsth: the original data of Joint PSTH
        """
        # Todo check data
        jpsth = [[[0 for _ in range(self.bin_number)] for _ in range(self.bin_number)]
                 for _ in range(len(self.reference_data))]

        for i in range(len(self.reference_data)):
            for j in range(len(self.psth_bottom_data[i])):
                for k in range(len(self.psth_select_data[i])):
                    # jpsth[i][j][k] means the data of (j,k) bin in ith trail
                    jpsth[i][j][k] = self.psth_bottom_data[i][j] * self.psth_select_data[i][k]

        return jpsth

    def get_processed_jpsth(self):
        """
        Returns
        -------
        processed_jpsth: The data processed by the original_jpsth
        """
        processed_jpsth = [[0 for _ in range(self.bin_number)] for _ in range(self.bin_number)]
        if self.normalization == 'Raw JPSTH':
            processed_jpsth = self._normalization_raw_jpsth()
        if self.normalization == 'JPSTH - PSTpred':
            processed_jpsth = self._normalization_jpsth_pstprod()
        if self.normalization == '(JPSTH-PSTHpred)/SDpred':
            processed_jpsth = self._normalization_jpsth_pstprod_sdpred()
        return processed_jpsth

    def _normalization_raw_jpsth(self):
        """
        Returns
        -------
        jpsth_norm: the data of raw Joint PSTH
        """
        jpsth_norm = [[0 for _ in range(self.bin_number)] for _ in
                      range(self.bin_number)]
        for i in range(len(self.reference_data)):
            for j in range(len(self.psth_bottom_data[i])):
                for k in range(len(self.psth_select_data[i])):
                    # jpsth_norm[j][k] means the data of (j,k) bin
                    jpsth_norm[j][k] += self.original_jpsth[i][j][k]
        return jpsth_norm

    def _normalization_jpsth_pstprod(self):
        """
        Returns
        -------
        jpsth_norm: the data of Joint PSTH-PSTpred
        """
        average_jpsth = self._get_average_jpsth()
        predict_jpsth = self._get_predict_jpsth()

        jpsth_norm = [[0 for _ in range(len(average_jpsth[i]))] for i in range(len(average_jpsth))]

        for i in range(len(jpsth_norm)):
            for j in range(len(jpsth_norm[i])):
                jpsth_norm[i][j] = average_jpsth[i][j] - predict_jpsth[i][j]

        return jpsth_norm

    def _normalization_jpsth_pstprod_sdpred(self):
        """
        Returns
        -------
        jpsth_norm: the data of (JPSTH - PSTH-PSTpred)/SDpred
        """
        jpsth_pstprod_ij = self._normalization_jpsth_pstprod()
        jpsth_pstprod_ii = JointPeriStimulusTimeHistogram(self.reference_data, self.bottom_data, self.bottom_data,
                                                          self.x_min, self.x_max,
                                                          self.bin_size,
                                                          normalization='JPSTH - PSTpred').get_processed_jpsth()
        jpsth_pstprod_jj = JointPeriStimulusTimeHistogram(self.reference_data, self.select_data, self.select_data,
                                                          self.x_min, self.x_max,
                                                          self.bin_size,
                                                          normalization='JPSTH - PSTpred').get_processed_jpsth()
        jpsth_norm = [[0 for _ in range(len(jpsth_pstprod_ij[i]))] for i in range(len(jpsth_pstprod_ij))]

        for i in range(len(jpsth_norm)):
            for j in range(len(jpsth_norm[i])):
                jpsth_norm[i][j] = jpsth_pstprod_ij[i][j] / math.sqrt(jpsth_pstprod_ii[i][i] * jpsth_pstprod_jj[j][j])

        return jpsth_norm

    def _get_psth(self, target_data):
        """
        Returns
        -------
        psth: the data of original PSTH
        """
        flag = 0
        psth = [[0 for _ in range(self.bin_number)] for _ in range(len(self.reference_data))]

        for i, event_time in enumerate(self.reference_data):
            for j in range(flag, len(target_data)):
                if self.x_min <= target_data[j] - event_time <= self.x_max:
                    flag = j
                    while self.x_min <= target_data[j] - event_time <= self.x_max:
                        psth[i][int((target_data[j] - event_time - self.x_min) / self.bin_size)] += 1
                        j = j + 1
                    break
        return psth

    def _get_average_psth(self, psth):
        """
        Calculate$$<n_{i}(u)> =\frac{1}{K}\sum_{k=1}^{K}{n_{i}^{(k)}(u)}$$
        i : Spike trains of neuron i.
        k : The k-th record (corresponding to the time point in reference_data).
        u : Data of the u-th bin.

        Returns
        -------
        psth: the data of average of PSTH
        """
        psth_sum = [0 for _ in range(len(psth[0]))]
        for i in range(len(psth)):
            for j in range(len(psth[i])):
                psth_sum[j] += psth[i][j]
        psth_sum = [number / len(psth) for number in psth_sum]
        return psth_sum

    def _get_average_jpsth(self):
        """
        Calculate $$<n_{ij}(u,v)> =\frac{1}{K}\sum_{k=1}^{K}{n_{ij}^{(k)}(u,v)}$$
        i, j: Spike trains of neuron i and neuron j.
        k   : The k-th record (corresponding to the time point in reference_data).
        u, v: Data at the (u, v) bin coordinates.

        Returns
        -------
        psth: the data of average of Joint PSTH
        """
        jpsth_sum = [[0 for _ in range(len(self.original_jpsth[0][i]))] for i in range(len(self.original_jpsth[0]))]
        for i in range(len(self.original_jpsth)):
            for j in range(len(self.original_jpsth[i])):
                for k in range(len(self.original_jpsth[i][j])):
                    jpsth_sum[j][k] += self.original_jpsth[i][j][k]

        for j in range(len(self.original_jpsth[0])):
            for k in range(len(self.original_jpsth[0][0])):
                jpsth_sum[j][k] /= len(self.original_jpsth)

        return jpsth_sum

    def _get_predict_jpsth(self):
        """
        Calculate $$\tilde{n}_{ij}(u,v)=<n_i(u)><n_j(v)>$$
        i, j: Spike trains of neuron i and neuron j.
        k   : The k-th record (corresponding to the time point in reference_data).
        u, v: Data at the coordinates of bin (u, v).

        Returns
        -------
        psth: the data of predict of Joint PSTH
        """
        average_psth_select = self._get_average_psth(self.psth_select_data)
        average_psth_bottom = self._get_average_psth(self.psth_bottom_data)

        predict_jpsth = [[0 for _ in range(len(average_psth_bottom))] for _ in range(len(average_psth_select))]

        for i in range(len(average_psth_bottom)):
            for j in range(len(average_psth_select)):
                predict_jpsth[i][j] = average_psth_bottom[i] * average_psth_select[j]

        return predict_jpsth


if __name__ == '__main__':
    select_name = 'datas/Neuron05b.txt'
    bottom_name = 'datas/Neuron04a.txt'
    reference_name = 'datas/Event04.txt'

    select = read_from_txt(select_name)
    bottom = read_from_txt(bottom_name)
    reference = read_from_txt(reference_name)

    test = JointPeriStimulusTimeHistogram(reference, select, bottom, x_min=-0.2, x_max=0.2,
                                          bin_size=0.05, normalization='(JPSTH-PSTHpred)/SDpred')
    # test.get_processed_jpsth()
    a = test.get_processed_jpsth()
    print()
