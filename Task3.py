import numpy as np
from matplotlib import pyplot as plt


def CorrelationDirect(input1, input2):
    len1 = len(input1)
    len2 = len(input2)
    length = len1 + len2 - 1
    y = np.zeros(length)

    for i in range(length):
        for j in range(len2):
            if 0 <= i - j < len1:
                y[i] += input1[i - j] * input2[j]
    return np.flip(-y, 0)


def CorrelationFFT(input1, input2):
    len1 = len(input1)
    len2 = len(input2)
    length = len1 + len2 - 1

    zeroes1 = np.zeros(length - len1)
    zeroes2 = np.zeros(length - len2)

    x_pad = np.concatenate([input1, zeroes1])
    y_pad = np.concatenate([zeroes2, input2])

    x_spec = np.fft.fft(x_pad)
    y_spec = np.fft.fft(y_pad)

    for i in range(length):
        tmp_re = x_spec.real[i] * y_spec.real[i] + x_spec.imag[i] * y_spec.imag[i]
        tmp_im = x_spec.imag[i] * y_spec.real[i] - x_spec.real[i] * y_spec.imag[i]
        x_spec.real[i] = tmp_re
        x_spec.imag[i] = tmp_im

    out = np.fft.ifft(x_spec)

    return out


# Quadro
T = 2
amplitude = 1
N_rect = 1000
sample_rate_rect = 100


def FuncRect(x):
    if x % T < 0.5 * T:
        return amplitude
    return -amplitude


def InitRectFunc(f):
    signal_ = list()
    time_arr = list()
    for i in range(N_rect):
        signal_.append(f(i / sample_rate_rect))
        time_arr.append(i / sample_rate_rect)
    return signal_, time_arr


rec_signal, time = InitRectFunc(FuncRect)
time_full = np.linspace(-N_rect//2//sample_rate_rect, N_rect//2//sample_rate_rect, num=N_rect)

kernel1 = list(map(lambda x: np.exp(-x ** 2), time_full))
kernel2 = list(map(lambda x: 2 * x if x < 1 else 0, time))

#####
fig, axs = plt.subplots(5)
axs[0].set_title("CorrelationNumpy Rect")
axs[0].plot(rec_signal)
axs[1].plot(kernel1)

numpy_corr = np.correlate(rec_signal, kernel1, mode='full')

axs[2].plot(numpy_corr/np.max(numpy_corr))
axs[3].plot(kernel2)

numpy_corr = np.correlate(rec_signal, kernel2, mode='full')

axs[4].plot(numpy_corr/np.max(numpy_corr))
plt.show()
######

fig, axs = plt.subplots(5)
axs[0].set_title("CorrelationDirect Rect")
axs[0].plot(rec_signal)
axs[1].plot(kernel1)

corr_direct1 = CorrelationDirect(rec_signal, kernel1)

axs[2].plot(corr_direct1/np.max(corr_direct1))
axs[3].plot(kernel2)

corr_direct2 = CorrelationDirect(rec_signal, kernel2)

axs[4].plot(corr_direct2/np.max(corr_direct2))
plt.show()

######
fig, axs = plt.subplots(5)
axs[0].set_title("CorrelationFFT Rect")
axs[0].plot(rec_signal)
axs[1].plot(kernel1)

corr_fft1 = CorrelationFFT(rec_signal, kernel1)

axs[2].plot(corr_fft1/np.max(corr_fft1))
axs[3].plot(kernel2)

corr_fft2 = CorrelationFFT(rec_signal, kernel2)

axs[4].plot(corr_fft2/np.max(corr_fft2))
plt.show()
