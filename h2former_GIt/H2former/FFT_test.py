import torch
import numpy as np

# 定义信号序列
N = 1000  # 信号长度
x = torch.linspace(0, 2*np.pi, N)
y = torch.sin(x) + 0.5*torch.sin(2*x) + 0.2*torch.sin(3*x) + torch.randn(N)*0.1  # 信号加入噪声

# 进行FFT变换
fft = torch.fft.fft(y)
real = fft.real
imag = fft.imag
# fft.# 进行快速傅里叶变换
freqs = torch.fft.fftfreq(N, d=1.0/(N-1))  # 计算频率

# 可视化结果
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1)
ax[0].plot(x, y)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')
ax[1].plot(freqs[:N//2], 20*torch.log10(torch.abs(fft[:N//2])/N))  # 转换为dB
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Magnitude (dB)')
plt.show()
