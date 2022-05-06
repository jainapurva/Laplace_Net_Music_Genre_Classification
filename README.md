# LaplaceNet
LaplaceNet: Deep Semi-Supervised Music Genre Classification 


## Setting Up Environment 
Requirements 
- faiss gpu 1.7.1
- pytorch 1.8.1
- cuda 10.2
- scipy 1.6.2
- tqdm 4.61.0
- numpy

From a clean conda enviroment you can perform the following commands to get a suitable enviroment
- conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
- conda install -c pytorch faiss-gpu 
- conda install -c anaconda scipy 
- conda install -c conda-forge tqdm 
- pip install torchnet 
- pip install matplotlib

Note that faiss-gpu has some compatibaility issues with certain versions of pytorch but the combination above is certified to work

## Running the program

> python main.py





