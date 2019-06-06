import numpy as np
import librosa

#https://www.kaggle.com/huseinzol05/sound-augmentation-librosa
def change_pitch(y, sr, pitch_change, bins_per_octave = 12):
    change = np.random.normal(0.0, pitch_change)
    return librosa.effects.pitch_shift(y, sr, n_steps=change, bins_per_octave=bins_per_octave)

def change_speed(y, sr, speed_change):
    change = np.random.normal(1.0, speed_change)
    if(change<0.6):
        change = 0.6
    return librosa.effects.time_stretch(y, change)

def add_noise(y, sr, sigma):
    return  y + np.random.normal(0.0, sigma, y.shape)

def combine(x1, x2, r1, r2):
    return np.multiply(r1,np.multiply( 1.0/np.max(np.abs(x1)) , x1 ))+np.multiply(r2,np.multiply( 1.0/np.max(np.abs(x2)) , x2 ))

def shift(y, sr, shift_sec, sigma=0.0005):
    shift_samples = shift_sec*sr
    if(shift_samples > 0):
        return np.concatenate((y, np.random.normal(0.0, sigma, np.abs(shift_samples))))
    if(shift_samples < 0):
        return np.concatenate((np.random.normal(0.0, sigma, np.abs(shift_samples)),y))
