import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

# Quadro
T = 2
amplitude = 1
N = 2
N_rect = 1000
sample_rate_rect = 100
# Cos
freq1 = 50
freq2 = 150
N_cos = 1000
sample_rate_cos = 10000


def f_rect(x):
    if x % T > 0.5 * T:
        return amplitude
    return -amplitude


def f_cos_150(x):
    return amplitude*np.cos(freq2 * x * 2 * np.pi)


def f_multi_cos(x):
    return 2*np.cos(freq1 * x * 2 * np.pi) + 1*np.cos(freq2 * x * 2 * np.pi)


def init_time_cos():
    time_arr = list()
    for i in range(N_cos):
        time_arr.append(i / sample_rate_cos)
    return time_arr


def init_cos_func(f):
    signal_ = list()
    for i in range(N_cos):
        signal_.append(f(i / sample_rate_cos))
    return signal_


def init_rect_func(f):
    signal_ = list()
    for i in range(N_rect):
        signal_.append(f(i / sample_rate_rect))
    return signal_


def init_time_rect():
    time_arr = list()
    for i in range(N_rect):
        time_arr.append(i / sample_rate_rect)
    return time_arr


def f_noise(x):
    if x % T > 0.5 * T:
        return amplitude + np.random.random(1)[0] / 2 * 5
    return -amplitude - np.random.random(1)[0] / 2 * 5


def f_cos_50(x):
    return 2*np.cos(freq1 * x * 2 * np.pi)


def f_cos_noise(x):
    return 2*np.cos(freq1 * x * 2 * np.pi) + np.random.random(1)[0] / 2 * 5


time_cos = init_time_cos()
cos_50 = init_cos_func(f_cos_50)
cos_150 = init_cos_func(f_cos_150)
multi = init_cos_func(f_multi_cos)
rect_signal = init_rect_func(f_rect)
time_rect = init_time_rect()

noise_cos_signal = init_cos_func(f_cos_noise)


def my_fftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    results = np.empty(n, int)
    N = (n-1)//2 + 1
    p1 = np.arange(0, N, dtype=int)
    results[:N] = p1
    p2 = np.arange(-(n//2), 0, dtype=int)
    results[N:] = p2

    return results * val


def DFT_slow(signal):
    N = len(signal)

    n = np.arange(N)
    k = np.reshape(n, (N, 1))

    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, signal)

    return X


X = DFT_slow(multi)
fourier = fft.fft(multi)

freq_arr = my_fftfreq(len(multi), 1/N_cos)

signal_phase = np.angle(X)

plt.title("Cos Signal")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.plot(time_cos, multi)
plt.show()

plt.title("Fourier scipy")
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.plot(np.abs(fourier))
plt.show()

plt.title("DFT_slow")
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.plot(freq_arr, np.abs(X))
plt.show()

plt.title("Noise Cos Signal")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.plot(time_cos, noise_cos_signal)
plt.show()

X_average = np.mean(noise_cos_signal)

new_signal = noise_cos_signal - X_average

X = DFT_slow(new_signal)

plt.title("DFT_slow_noise")
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.xlim(-10, 10)
plt.plot(freq_arr, np.abs(X))
plt.show()

X = DFT_slow(cos_50)
plt.title("DFT_slow_clear")
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.plot(freq_arr, np.abs(X))
plt.show()

X = DFT_slow(rect_signal)

signal_phase = np.angle(X)[:len(X)]

plt.title("Signal")
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.plot(time_rect, rect_signal)
plt.show()

plt.title("DFT_slow")
plt.ylabel("amplitude")
plt.plot(np.abs(X))
plt.show()
