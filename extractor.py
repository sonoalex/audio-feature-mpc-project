import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np


ocean_sound = 'big-waves-hit-land.wav'
dolly_box = 'dollybox.wav'
toy = 'percussion_toy.wav'

# load audio files with librosa
scale_ocean, sr_ocean = librosa.load(ocean_sound)
scale_dolly, sr_dolly = librosa.load(dolly_box)
scale_toy, sr_toy = librosa.load(toy)

mel_spectrogram_ocean = librosa.feature.melspectrogram(scale_ocean, sr=sr_ocean, n_fft=2048, hop_length=512, n_mels=10)
log_mel_spectrogram_ocean = librosa.power_to_db(mel_spectrogram_ocean)



plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram_ocean, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr_ocean)
plt.colorbar(format="%+2.f")

mel_spectrogram_dolly = librosa.feature.melspectrogram(scale_dolly, sr=sr_dolly, n_fft=2048, hop_length=512, n_mels=10)
log_mel_spectrogram_dolly = librosa.power_to_db(mel_spectrogram_dolly)


plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram_dolly, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr_dolly)
plt.colorbar(format="%+2.f")

mel_spectrogram_toy = librosa.feature.melspectrogram(scale_toy, sr=sr_toy, n_fft=2048, hop_length=512, n_mels=10)
log_mel_spectrogram_toy = librosa.power_to_db(mel_spectrogram_toy)


plt.figure(figsize=(25, 10))
plt.title('Percussion toy sound')
librosa.display.specshow(log_mel_spectrogram_toy, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr_toy)
plt.colorbar(format="%+2.f")



cent = librosa.feature.spectral_centroid(y=scale_ocean, sr=sr_ocean)
S, phase = librosa.magphase(librosa.stft(y=scale_ocean))
librosa.feature.spectral_centroid(S=S)
times = librosa.times_like(cent)
fig, ax = plt.subplots()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax)
ax.plot(times, cent.T, label='Spectral centroid', color='w')
ax.legend(loc='upper right')
ax.set(title='Ocean spectrogram')

cent = librosa.feature.spectral_centroid(y=scale_dolly, sr=sr_dolly)
S, phase = librosa.magphase(librosa.stft(y=scale_dolly))
librosa.feature.spectral_centroid(S=S)
times = librosa.times_like(cent)
fig, ax = plt.subplots()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax)
ax.plot(times, cent.T, label='Spectral centroid', color='w')
ax.legend(loc='upper right')
ax.set(title='Dolly spectrogram')

cent = librosa.feature.spectral_centroid(y=scale_toy, sr=sr_toy)
S, phase = librosa.magphase(librosa.stft(y=scale_toy))
librosa.feature.spectral_centroid(S=S)
times = librosa.times_like(cent)
fig, ax = plt.subplots()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax)
ax.plot(times, cent.T, label='Spectral centroid', color='w')
ax.legend(loc='upper right')
ax.set(title='Toy spectrogram')

plt.show()