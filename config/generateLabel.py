import random
import os
import shutil

base = os.getcwd()
label_dir = base+'/data-local/labels/gtzan/'
data_dir = base+'/data-local/images/gtzan/by-image/train/'
#print(os.listdir(data_dir))
with open(label_dir+'/10.txt','w') as reader:
    for folderLoc in os.listdir(data_dir):
        fileList = os.listdir(data_dir+'/'+folderLoc)
        random.shuffle(fileList)
        for i in fileList[:30]:
            reader.write(i+' '+folderLoc+'\n')

    