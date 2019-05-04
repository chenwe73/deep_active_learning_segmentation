import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import entr
import cv2
import ntpath
import os
import random

import tensorflow as tf
from keras import backend as K

import Models , LoadBatches
from MeanIoU import MeanIoU
from constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS

sess = tf.Session()
K.set_session(sess)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.tensorflow_backend.set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str)
parser.add_argument("--list_file", type = str)
parser.add_argument("--model_name", type = str, default = "vgg_segnet")
parser.add_argument("--n_classes", type=int, default = 19)
parser.add_argument("--input_height", type=int , default = 224 )
parser.add_argument("--input_width", type=int , default = 224 )
parser.add_argument("--seed", type=int , default = 123 )
args = parser.parse_args()

is_plot = True # False


def entropy(model, X, img_name):
    predict = model.predict(X)[0]
    entropy = entr(predict).sum(axis=1)/np.log(2)
    
    if (is_plot):
        entropy_map = np.reshape(entropy, (model.outputWidth, model.outputHeight))
        plt.imshow(entropy_map)
        plt.colorbar()
        directory = "./tmp/entropy/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + img_name)
        plt.close()
    
    acq = np.mean(entropy, axis=0)
    return acq


def BALD(model, X, img_name):
    nb_MC_samples = 100
    MC_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    learning_phase = True  # use dropout at test time
    MC_samples = [MC_output([X, learning_phase])[0] for _ in range(nb_MC_samples)]
    MC_samples = np.array(MC_samples)  # [#samples x batch size x #classes]
    
    expected_entropy = - np.mean(np.sum(MC_samples * np.log(MC_samples + 1e-10), axis=-1), axis=0)  # [batch size] # mean of entropy of each pixel
    expected_p = np.mean(MC_samples, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size] # entropy of mean of each pixel
    BALD_acq = entropy_expected_p - expected_entropy
    
    if (is_plot):
        BALD_map = BALD_acq[0].reshape(( model.outputWidth ,  model.outputHeight ) )
        plt.imshow(BALD_map)
        plt.colorbar()
        directory = "./tmp/bald/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + img_name)
        plt.close()
    
    acq = np.mean(BALD_acq[0])
    return acq


def inference(model, X, img_name):
    colors = np.array(CITYSCAPES_LABEL_COLORS)
    colors[:,[2, 0]] = colors[:,[0, 2]] #RGB2BGR
    
    predict = model.predict(X)[0]
    pr = predict.reshape(( model.outputWidth ,  model.outputHeight , args.n_classes ))
    pr = pr.argmax( axis=2 )
    #print(np.unique(pr))
    print(np.sum(X))
    print(np.sum(pr))
    
    seg_img = np.zeros( ( model.outputWidth ,  model.outputHeight , 3 ) )
    for c in range(args.n_classes):
        seg_img[:,:,0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    #seg_img = cv2.resize(seg_img  , (args.input_width , args.input_height ))
    directory = "./tmp/inference/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(directory + img_name, seg_img)


def main():
    random.seed(args.seed)
    
    # model
    modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 
        'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 
        'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
    modelFN = modelFns[ args.model_name ]
    
    model = modelFN( args.n_classes , input_height=args.input_height, input_width=args.input_width   )
    #model = deeplab.Deeplabv3(input_shape=(512, 512, 3), classes=args.n_classes)
    
    model.load_weights(args.save_weights_path)
    miou_metric = MeanIoU(args.n_classes)
    model.compile(loss='categorical_crossentropy',
          optimizer= 'adam' ,
          metrics=['accuracy', miou_metric.mean_iou])
    
    # open list_file
    with open(args.list_file) as f:
        list_line = f.readlines()
    content = [x.strip().split(",") for x in list_line]
    images, segmentations = zip(*content)
    list_size = len(images)
    print(list_size)
    
    # inference
    acq = np.zeros(list_size)
    for i in range(list_size):
        img_name = ntpath.basename(images[i])
        imgArr = LoadBatches.getImageArr( images[i] , args.input_width , args.input_height )
        X = np.array([imgArr])
        segArr = LoadBatches.getSegmentationArr( segmentations[i] , 
            args.n_classes, model.outputWidth , model.outputHeight )
        Y = np.array([segArr])
        
        evaluation = model.evaluate(x=X, y=Y, batch_size=1, verbose=0)
        acc = evaluation[1]
        miou = evaluation[2]
        
        # acquisition function
        #acq[i] = random.random()
        acq[i] = entropy(model, X, img_name)
        #acq[i] = BALD(model, X, img_name)
        #acq[i] = -1 * acc
        #acq[i] = -1 * miou
        
        if (is_plot):
            inference(model, X, img_name)
    
    # sort list_file
    sort = sorted(zip(acq, list_line))
    acq_sorted, list_sorted = zip(*sort)
    list_sorted = list_sorted[::-1]
    acq_sorted = acq_sorted[::-1]
    print(acq_sorted)
    
    if (False):
        plt.hist(acq)
        plt.savefig("./tmp/acq_hist.png")
        plt.close()
    
    with open(args.list_file, 'w') as f:
        for i in list_sorted:
            f.write("%s" % i)
    
    
    
main()



    
'''
with open('./tmp/acq', 'w') as f:
    for i in acq:
        f.write("%s\n" % i)

plt.hist(entropy_image)
plt.savefig("./tmp/entropy_dist.png")
plt.close()

entropy_class = np.zeros((list_size, args.n_classes))
for i in range(args.n_classes):
    mask = np.logical_not((label == i))
    masked = np.ma.array(entropy, mask = mask)
    entropy_class[:,i] = masked.mean(axis=1)
    
entropy_class_all = entropy_class.mean(axis=0)
print(np.shape(entropy_class_all))
for i in range(len(entropy_class_all)):
    print(entropy_class_all[i])

mean_entropy_class = entropy_class.mean(axis=1)
print(np.shape(mean_entropy_class))
plt.hist(mean_entropy_class)
plt.savefig("./tmp/mean_entropy_class.png")
plt.close()
'''
    



