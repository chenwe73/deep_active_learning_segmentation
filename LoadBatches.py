import numpy as np
import cv2
import glob
import itertools
import os
import scipy.misc



def getImageArr( path , width , height , imgNorm="divide" , odering='channels_first' ):

    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width , height ), interpolation=cv2.INTER_NEAREST)
    #cv2.imwrite(  "./tmp/resize.png" , img )
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
    #np.savetxt('./tmp/img.out', img, delimiter='', fmt='%3u')
    
    seg_labels = np.zeros((  height , width  , nClasses ))
    # out of nClasses will default to 0. add another class? TODO
    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)
    
    # 1 class
    #seg_labels[: , : , 0 ] = (img == 0 ).astype(int)
    #seg_labels[: , : , 1 ] = (img != 0 ).astype(int)
    
    
    #label = seg_labels.argmax(axis=2)
    #scipy.misc.imsave("./tmp/label.png", label)
    '''
    img_name = path.split('/')[-1].split(".")[0]
    directory = "./tmp/seg/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    scipy.misc.imsave(directory+img_name+".png", label)
    #print(np.unique(label))
    #np.savetxt("./tmp/"+img_name+".map", label, delimiter='', fmt='%3u')
    '''
    
    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    
    return seg_labels



def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width , list_file=None  ):
    
    #assert images_path[-1] == '/'
    #assert segs_path[-1] == '/'
    
    cityscapes = 1
    _DEFAULT_PATTEN = {
        'input': '*_leftImg8bit.png',
        'label': '*_gtFine_labelTrainIds.png',
    }
    
    if (list_file):
        with open(list_file) as f:
            content = f.readlines()
        content = [x.strip().split(",") for x in content]
        images, segmentations = zip(*content)
    else:
        if (cityscapes):
            images_path = os.path.join(images_path, '*', _DEFAULT_PATTEN['input'])
            segs_path = os.path.join(segs_path, '*', _DEFAULT_PATTEN['label'])
            images = glob.glob(images_path)
            segmentations = glob.glob(segs_path)
        else:
            images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
            segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
        images.sort()
        segmentations.sort()
    
    print(len(images))

    assert len( images ) == len(segmentations)
    #for im , seg in zip(images,segmentations):
    #    assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

    zipped = itertools.cycle( zip(images,segmentations) )

    while True:
        X = []
        Y = []
        for _ in range( batch_size) :
            im , seg = next(zipped)
            #print(im, seg)
            imgArr = getImageArr(im , input_width , input_height )
            segArr = getSegmentationArr( seg , n_classes , output_width , output_height )
            #imgArr = np.zeros([3, input_width , input_height])
            #segArr = np.zeros([output_height*output_width,19])
            X.append(imgArr)
            Y.append(segArr)

        yield np.array(X) , np.array(Y)


