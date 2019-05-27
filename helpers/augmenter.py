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

def combine(y1, y1_mul, y2, y2_mul):
    max_len = max(y1.shape[0], y2.shape[0])
    #extend shorter recording
    if(y1.shape[0]<max_len):
        tmp_y1 = np.concatenate((y1,np.zeros(max_len-y1.shape[0])))
    else:
        tmp_y1 = y1
        
    if(y2.shape[0]<max_len):
        tmp_y2 = np.concatenate((y2,np.zeros(max_len-y2.shape[0])))
    else:
        tmp_y2 = y2
    return y1_mul*tmp_y1 + y2_mul*tmp_y2

def shift(y, sr, shift_sec, sigma=0.0005):
    shift_samples = shift_sec*sr
    if(shift_samples > 0):
        return np.concatenate((y, np.random.normal(0.0, sigma, np.abs(shift_samples))))
    if(shift_samples < 0):
        return np.concatenate((np.random.normal(0.0, sigma, np.abs(shift_samples)),y))
