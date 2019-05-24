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
