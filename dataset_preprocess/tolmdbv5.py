import os    
import lmdb # install lmdb by "pip install lmdb"
import cv2
import re
from PIL import Image
import numpy as np
import imghdr
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_txt', type=str, default='', help='name and labels of your training set')
parser.add_argument('--save_path', type=str, default='', help='the path to save your lmdb data')
opt = parser.parse_args()
imageBin=0

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype='uint8')
        #imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        print(imgH + imgW)
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False		
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            #txn.put(k, v)
            txn.put(str(k).encode(), str(v).encode())
			
def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    global imageBin
    for i in range(nSamples):  
    #for i in range(4):  
        print(imagePathList[i]+"imagePathList[i]") 
        imagePath = ''.join(imagePathList[i]).split(",")[0].replace('\n','').replace('\r\n','')
        print(imagePath+" imagePath captured")
        label = ''.join(labelList[i])
        print(label+" label captured")
        #if not os.path.exists(imagePath):
        #    print('%s does not exist' % imagePath)
        #    continue	
        script_path=os.path.abspath(__file__)
        script_dir=os.path.split(script_path)[0]
        relpath="images/"+imagePath
        abs_file_path=os.path.join(script_dir,relpath)
        print(abs_file_path)
        #imageBin = Image.open(imagePath)
        #print(imageBin)
        try:
            #with open(abs_file_path, 'rb') as f:
            #with open('./images/'+imagePath, 'r') as f:
                #imageBin = f.read()
            with open(abs_file_path, 'rb') as image_stream:
                imageBin = image_stream.read()
            #imageBin = Image.open(imagePath)
            print(imageBin)
        except Exception as e:
            print(imagePath+" try_except exception")  

        if imageBin is None:
            print("imageBin None") 
        try:
            #imageBuf = np.fromstring(imageBin, dtype='uint8')
            imageBuf = np.fromstring(imageBin, dtype=np.uint8)
            #print(imageBuf)
            img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
            imgH = int(img.shape[0])
            imgW = int(img.shape[1])
            #imgH, imgW = img.shape[0], img.shape[1]
            print(imgH + imgW  )
            print( imgH  )
            print( imgW  )
        except:
            print("except-imageBin imageBuf")
        else:
            if imgH * imgW == 0:
                print("imgH*imgW==0")		

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        #if cnt % 1000 == 0:
        if cnt % 100 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
        print(cnt)
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
	

if __name__ == '__main__':

    outputPath = opt.save_path
    imgdata = open(opt.data_txt)
    imagePathList = list(imgdata)
    
    labelList = []
    for line in imagePathList:
        line = line.strip().strip('\n')
    	# Ensure that you are not working on empty line
        if line:
       	        word  = line.split(",") 
    	# Ensure that index is not out of range
        if len(word) > 1: 
            #print(word[1])
            labelList.append(word[1]) 
            #word = line.split(",")[1]
            #print(word)
            #print(word)
            #labelList.append(word[1])
    createDataset(outputPath, imagePathList, labelList)
    #pass 2
