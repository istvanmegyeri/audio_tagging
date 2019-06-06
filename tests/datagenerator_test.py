from helpers.data_downloader import *
from helpers.datageneratormemory import *
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


downloader = data_downloader("1_oY3xeJsaZOXqikNn1aAbiX4SOZ2KZHR", "data.zip") #curated compressed
full_data = downloader.load_compressed_windows('F:')

labels = [0] * 4970
dgm = DataGeneratorMemory(full_data,labels,batch_size=10 )
cnt = 0
sr = 22050
specs, y = dgm.__getitem__(2)
print(specs.shape)
librosa.display.specshow(specs[4,:,:,0], hop_length=1024)
print(y[0,:])

