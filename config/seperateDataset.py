import os
import random
import shutil

base = '/Users/apurva/Data/Pattern Recognition/PRLaplaceStat'
data_dir = base+'/data-local/images/gtzan/data/'
test_dir = base+'/data-local/images/gtzan/by-image/test/'
trainVal_dir = base+'/data-local/images/gtzan/by-image/train+val/'
val_dir = base+'/data-local/images/gtzan/by-image/val/'
train_dir = base+'/data-local/images/gtzan/by-image/train/'
#print(os.listdir(data_dir))
for folderLoc in os.listdir(data_dir):
    #print(folderLoc)
    indexes = list(range(100))
    random.shuffle(indexes)
    genre = os.path.basename(folderLoc)
    # Seperate indexes for train
    for i in indexes[:55]:
        filename = genre+str(i).rjust(5,'0')+'.png'
        src = data_dir+'/'+genre+'/'+filename
        dest = train_dir+'/'+genre+'/'
        dest1 = trainVal_dir+'/'+genre+'/'
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        os.makedirs(os.path.dirname(dest1), exist_ok=True)
        shutil.copy2(src, dest1)
    # Seperate indexed for validation
    for i in indexes[55:70]:
        filename = genre+str(i).rjust(5,'0')+'.png'
        src = data_dir+'/'+genre+'/'+filename
        dest = val_dir+'/'+genre+'/'
        dest = trainVal_dir+'/'+genre+'/'
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        os.makedirs(os.path.dirname(dest1), exist_ok=True)
        shutil.copy2(src, dest1)
    # Seperate indexes for test
    for i in indexes[70:]:
        filename = genre+str(i).rjust(5,'0')+'.png'
        src = data_dir+'/'+genre+'/'+filename
        dest = test_dir+'/'+genre+'/'
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)