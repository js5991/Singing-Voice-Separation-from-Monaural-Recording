
import shutil
import os
import random

source = '../data/'
dest1 = '../data/train/'
dest2 = '../data/valid/'

wavfiles = []
for (root, dirs, files) in os.walk(self.path):
    wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])
    
random.shuffle(wavfiles)
train = wavfiles[:800]
valid = wavfiles[:200]

try:
    os.makedirs(dest1); ## it creates the destination folder
    os.makedirs(dest2);
except:
    print ("Folder already exist or some error")

for file in train:
    shutil.copy2(file, dest1)
for file in valid:
    shutil.copy2(file, dest2)    