import numpy as np
import cv2
import glob
import itertools
import os
import scipy.misc
import ntpath
import matplotlib.pyplot as plt


def getImageArr( path , width , height , imgNorm="divide" , odering='channels_first' ):

    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width , height ), interpolation=cv2.INTER_NEAREST)
    #cv2.imwrite(  "resize.png" , img )
    img = img.astype(np.float32)
    img = img / 255.0
    if odering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img



def getSegmentationArr( path , nClasses ,  width , height  ):

    img = cv2.imread(path, 1)
    #print(np.unique(img))
    img = cv2.resize(img, ( width , height ), interpolation=cv2.INTER_NEAREST) # must use INTER_NEAREST
    img = img[:, : , 0]
    #np.savetxt('img.out', img, delimiter='', fmt='%3u')
    
    seg_labels = np.zeros((  height , width  , nClasses ))
    # out of nClasses will default to 0. add another class? TODO
    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)
    
    # 1 class
    #seg_labels[: , : , 0 ] = (img == 0 ).astype(int)
    #seg_labels[: , : , 1 ] = (img != 0 ).astype(int)
    
    '''
    label = seg_labels.argmax(axis=2)
    img_name = path.split('/')[-1].split(".")[0]
    directory = "./tmp/seg/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    scipy.misc.imsave(directory+img_name+".jpg", label)
    #print(np.unique(label))
    #np.savetxt("./tmp/"+img_name+".map", label, delimiter='', fmt='%3u')
    '''
    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    
    return seg_labels

def main():
    list_file = "./out/pool.csv"
    
    input_width = 224
    input_height = 224
    outputWidth = 224
    outputHeight = 224
    n_classes = 19
    
    # open list_file
    with open(list_file) as f:
        list_line = f.readlines()
    content = [x.strip().split(",") for x in list_line]
    images, segmentations = zip(*content)
    list_size = len(images)
    print(list_size)
    
    # inference
    data = np.empty((0, n_classes))
    for i in range(list_size):
        print(i)
        img_name = ntpath.basename(images[i])
        #imgArr = getImageArr( images[i] , input_width , input_height )
        #X = np.array([imgArr])
        segArr = getSegmentationArr( segmentations[i] , 
            n_classes, outputWidth , outputHeight )
        Y = np.array([segArr])
        
        label = Y.argmax(axis=2)
        unique = np.unique(label)
        len_unique = len(unique)
        
        onehot = np.zeros(n_classes)
        onehot[unique] = 1
        
        data = np.append(data, [onehot], axis=0)
        
    
    data = np.sum(data, axis=0)
    print (data)
    
    plt.bar(np.arange(len(data)), data)
    plt.savefig("./tmp/classes.png")
    plt.close()

main()




