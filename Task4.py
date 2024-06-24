import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def MovingAverageFilter_1(signal_in, window_size):
    filter_coeffs = np.zeros(len(signal_in))
    filter_coeffs[:window_size//2] = 2/window_size
    filter_coeffs[-window_size//2:] = 2/window_size
    output = np.convolve(signal_in, filter_coeffs)
    return output


def MovingAverageFilter_2(signal_in, window_size):
    filter_coeffs = np.zeros(len(signal_in))
    filter_coeffs[:window_size] = 1/window_size
    output = np.convolve(signal_in, filter_coeffs)
    return output


N = 1000

time = np.linspace(0, 2*np.pi, N)
noise = np.random.normal(0, 0.5, N)

signal_rect = signal.square(2 * np.pi * time)
signal_cos = np.cos(2 * np.pi * time)

signal_rect_noise = signal.square(2 * np.pi * time) + noise


window_size = 50
filter_coeffs = np.zeros(len(signal_rect_noise))
filter_coeffs[:window_size // 2] = 2 / window_size
filter_coeffs[-window_size // 2:] = 2 / window_size
filter_coeffs2 = np.zeros(len(signal_rect_noise))
filter_coeffs2[:window_size] = 1 / window_size
plt.plot(np.angle(np.fft.fft(filter_coeffs)))
plt.plot(np.angle(np.fft.fft(filter_coeffs2)))
plt.show()

filtered_signal0 = MovingAverageFilter_1(signal_rect_noise, 4)
filtered_signal1 = MovingAverageFilter_1(signal_rect_noise, 11)
filtered_signal2 = MovingAverageFilter_1(signal_rect_noise, 50)

filtered_signal3 = MovingAverageFilter_2(signal_rect_noise, 4)
filtered_signal4 = MovingAverageFilter_2(signal_rect_noise, 11)
filtered_signal5 = MovingAverageFilter_2(signal_rect_noise, 50)

plt.plot(filtered_signal2)
plt.plot(filtered_signal5)
plt.show()


fig, axs = plt.subplots(5)
axs[0].plot(signal_rect_noise[:1000], label='Original')
axs[0].legend()
axs[1].plot(filtered_signal1[:1000], label='WindowSize 11')
axs[1].legend()
axs[2].plot(filtered_signal2[:1000], label='WindowSize 51')
axs[2].legend()
axs[3].plot(signal_rect, label='Clear')
axs[3].legend()
axs[4].plot(filtered_signal0[:1000], label='WindowSize 4')
axs[4].legend()
plt.show()

fig, axs = plt.subplots(5)
axs[0].plot(signal_rect_noise[:1000], label='Original')
axs[0].legend()
axs[1].plot(filtered_signal3[:1000], label='WindowSize 11')
axs[1].legend()
axs[2].plot(filtered_signal4[:1000], label='WindowSize 51')
axs[2].legend()
axs[3].plot(signal_rect, label='Clear')
axs[3].legend()
axs[4].plot(filtered_signal5[:1000], label='WindowSize 4')
axs[4].legend()
plt.show()

for window_size in range(7, 51, 5):
    w = np.linspace(0, 500, N//2)
    h = np.append(np.ones(window_size)/window_size, np.zeros(N - window_size))
    H = np.fft.fft(h)
    for i in range(0, len(H)):
        H[i] = H[i]*(N//2)
    plt.plot(w, np.abs(H[0: len(H) // 2]), label='Размер окна = ' + str(window_size))
plt.xlabel("Частота")
plt.ylabel("Амплитуда")
plt.legend()
plt.show()


for window_size in range(7, 51, 7):
    phase_response = np.angle(np.fft.fft(np.ones(window_size), 1024))
    phase_response = np.unwrap(phase_response)
    plt.plot(np.arange(len(phase_response)), phase_response)
plt.title("Фазовая характеристика фильтра скользящего среднего")
plt.xlabel("Частота")
plt.ylabel("Фаза")
plt.grid()
plt.show()

filtered_signal0 = MovingAverageFilter_1(signal_rect_noise, 4)
filtered_signal1 = MovingAverageFilter_1(signal_rect_noise, 11)
filtered_signal2 = MovingAverageFilter_1(signal_rect_noise, 51)

fig, axs = plt.subplots(5)
axs[0].plot(signal_rect_noise, label='Original')
axs[0].legend()
axs[1].plot(filtered_signal1[:1000], label='WindowSize 11')
axs[1].legend()
axs[2].plot(filtered_signal2[:1000], label='WindowSize 51')
axs[2].legend()
axs[3].plot(signal_rect, label='Clear')
axs[3].legend()
axs[4].plot(filtered_signal0[:1000], label='WindowSize 4')
axs[4].legend()
plt.show()

fft_signal = np.fft.fft(signal_rect_noise)
fft_filtered_1 = np.fft.fft(filtered_signal1)
fft_filtered_2 = np.fft.fft(filtered_signal2)
fft_clean = np.fft.fft(signal_rect)

signal_cos_noise = signal_cos + noise
filtered_signal1 = MovingAverageFilter_1(signal_cos_noise, 11)
filtered_signal2 = MovingAverageFilter_1(signal_cos_noise, 51)

fig, axs = plt.subplots(4)
axs[0].plot(signal_cos_noise, label='Original')
axs[0].legend()
axs[1].plot(filtered_signal1[:1000], label='WindowSize 11')
axs[1].legend()
axs[2].plot(filtered_signal2[:1000], label='WindowSize 51')
axs[2].legend()
axs[3].plot(signal_cos, label='Clear')
axs[3].legend()
plt.show()