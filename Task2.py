import numpy as np
from matplotlib import pyplot as plt


def ConvolutionDirect(input1, input2):
    len1 = len(input1)
    len2 = len(input2)
    length = len1 + len2 - 1
    y = np.zeros(length)

    for i in range(length):
        for j in range(len1):
            if 0 <= i - j < len2:
                y[i] += input1[j] * input2[i - j]
    return y


def ConvolutionFFT(input1, input2):
    len1 = len(input1)
    len2 = len(input2)
    length = len1 + len2 - 1

    x_pad = np.pad(input1, (0, length - len1))
    y_pad = np.pad(input2, (0, length - len2))

    x_spec = np.fft.fft(x_pad)
    y_spec = np.fft.fft(y_pad)

    for i in range(length):
        tmp_re = x_spec.real[i]*y_spec.real[i] - x_spec.imag[i]*y_spec.imag[i]
        tmp_im = x_spec.imag[i]*y_spec.real[i] + x_spec.real[i]*y_spec.imag[i]
        x_spec.real[i] = tmp_re
        x_spec.imag[i] = tmp_im
    out = np.fft.ifft(x_spec)
    return out


# Cos
amplitude_cos = 1
freq1 = 50
freq2 = 150
N_cos = 1000
sample_rate_cos = 10000


def Cos50(x):
    return amplitude_cos*np.cos(freq1 * x * 2 * np.pi)


def Cos150(x):
    return amplitude_cos*np.cos(freq2 * x * 2 * np.pi)


def MultiCos(x):
    return amplitude_cos*np.cos(freq1 * x * 2 * np.pi) + amplitude_cos*np.cos(freq2 * x * 2 * np.pi)


def TimeCos():
    time_arr = list()
    for i in range(N_cos):
        time_arr.append(i / sample_rate_cos)
    return time_arr


def CosSignal(f):
    signal_ = list()
    for i in range(N_cos):
        signal_.append(f(i / sample_rate_cos))
    return signal_


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

kernel1 = list(map(lambda x: np.exp(-x ** 2 * 20), time_full))
kernel2 = list(map(lambda x: 2 * x if x < 1 else 0, time))

fig, axs = plt.subplots(5)
axs[0].set_title("Numpy Convolve Rect")
axs[0].plot(rec_signal)
axs[1].plot(kernel1)
axs[2].plot(np.convolve(rec_signal, kernel1))
axs[3].plot(kernel2)
axs[4].plot(np.convolve(rec_signal, kernel2))
plt.show()

fig, axs = plt.subplots(5)
axs[0].set_title("ConvolutionDirect Rect")
axs[0].plot(rec_signal)
axs[1].plot(kernel1)
conv_direct1 = ConvolutionDirect(rec_signal, kernel1)
axs[2].plot(conv_direct1/np.max(conv_direct1))
axs[3].plot(kernel2)
conv_direct1 = ConvolutionDirect(rec_signal, kernel2)
axs[4].plot(conv_direct1/np.max(conv_direct1))
plt.show()

fig, axs = plt.subplots(5)
axs[0].set_title("ConvolutionFFT Rect")
axs[0].plot(rec_signal)
axs[1].plot(kernel1)
conv_fft1 = ConvolutionFFT(rec_signal, kernel1)
axs[2].plot(conv_fft1/np.max(conv_fft1))
axs[3].plot(kernel2)
conv_fft1 = ConvolutionFFT(rec_signal, kernel2)
axs[4].plot(conv_fft1/np.max(conv_fft1))
plt.show()


np_conv = np.convolve(rec_signal, kernel2)

conv_direct1 = ConvolutionDirect(rec_signal, kernel2)
conv_fft1 = ConvolutionFFT(rec_signal, kernel2)
plt.title("Comparing Numpy and Our")
plt.plot(conv_direct1/np.max(conv_direct1), color='r', label='Direct')
plt.plot(conv_fft1/np.max(conv_fft1), color='g', label='FFT')
result = np_conv - conv_direct1
plt.plot(result, color='b', label='Numpy - direct')
plt.legend()
plt.show()

result1 = np_conv - conv_fft1
plt.plot(result1, color='g', label='Numpy - fft')
plt.legend()
plt.show()

cos_50 = CosSignal(Cos50)
cos_150 = CosSignal(Cos150)
time_cos = TimeCos()

conv_fft = ConvolutionFFT(cos_50, kernel1)
conv_direct = ConvolutionDirect(cos_50, kernel1)

fig, axs = plt.subplots(5)
axs[0].set_title("ConvolutionFFT/Direct Cos_50")
axs[0].plot(cos_50)
axs[1].plot(kernel1)
axs[2].plot(conv_fft/np.max(conv_fft))
axs[3].plot(kernel2)
axs[4].plot(conv_direct/np.max(conv_direct))
plt.show()

noise = np.random.normal(size=len(rec_signal))/10
noise1 = np.random.normal(size=len(rec_signal))/2
noise2 = np.random.random(size=len(rec_signal))/2

noise_signal = rec_signal + noise
kernel_noise = kernel1 + + noise

noise_signal2 = rec_signal + noise1
kernel_noise2 = kernel1 + noise1

kernel_noise3 = kernel1 + noise2

conv_fft1 = ConvolutionFFT(noise_signal, kernel_noise)
conv_direct1 = ConvolutionDirect(noise_signal, kernel_noise)

conv_direct2 = ConvolutionDirect(noise_signal2, kernel_noise)
conv_fft2 = ConvolutionFFT(noise_signal2, kernel_noise2)

conv_fft3 = ConvolutionFFT(noise_signal2, kernel_noise2)

fig, axs = plt.subplots(9)
axs[0].set_title("ConvolutionFFT/Direct Noise Rect")
axs[0].plot(noise_signal)
axs[1].plot(kernel_noise)

axs[2].plot(conv_fft1/np.max(conv_fft1))
axs[3].plot(conv_direct1/np.max(conv_direct1))

axs[4].plot(noise_signal2)
axs[5].plot(conv_fft2/np.max(conv_fft2))
axs[6].plot(conv_direct2/np.max(conv_direct2))

axs[7].plot(kernel_noise2)

axs[8].plot(conv_fft3/np.max(conv_fft3))
plt.show()

sample_rate_rect = 1000
T = 0.02 * np.pi
rec, time = InitRectFunc(FuncRect)
np_conv = np.correlate(cos_150, rec, mode='full')
convolve = ConvolutionFFT(cos_150, rec)

plt.plot(-np_conv)
plt.plot(convolve)
plt.show()
