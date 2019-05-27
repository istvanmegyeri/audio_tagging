#import
import os

class data_downloader:
  def __init__(self, file_id, filename):
    self.file_id = file_id
    self.filename = filename
    
  def download(self):
    bashCommand = "wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id="+self.file_id+"' -O "+self.filename
    os.system(bashCommand)
    
  def unzip(self):
    !mkdir -p data
    bashCommand = "unzip "+self.filename+" -d data/"
    os.system(bashCommand)
    
  def delete(self):
    bashCommand = "rm -f "+self.filename
    os.system(bashCommand)
    
  def load_compressed(self):
    full_data = []
    for i in range(1,11):
      temp = np.load('data/data_c'+str(i)+'.npz')
      B= np.split(temp['arr_0'], np.where(temp['arr_0'][:]== 200)[0][1:])
      for wave in B:
        if(wave.size > 10):
          full_data.append(wave[1:]) 
        
    return full_data
  
#downloader = data_downloader("1_oY3xeJsaZOXqikNn1aAbiX4SOZ2KZHR", "data.zip") #curated compressed
#downloader.download()
#downloader.unzip()
#downloader.delete()
