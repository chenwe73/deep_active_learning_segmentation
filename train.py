import argparse
import Models , LoadBatches
import pickle

from MeanIoU import MeanIoU

import tensorflow as tf
from keras import backend as K
from keras import optimizers

from Models import deeplab

sess = tf.Session()
K.set_session(sess)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.tensorflow_backend.set_session(tf.Session(config=config))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_weights_path", type = str )
    parser.add_argument("--train_images", type = str )
    parser.add_argument("--train_annotations", type = str )
    parser.add_argument("--n_classes", type=int )
    parser.add_argument("--input_height", type=int , default = 224 )
    parser.add_argument("--input_width", type=int , default = 224 )

    parser.add_argument('--validate',action='store_false')
    parser.add_argument("--val_images", type = str , default = "")
    parser.add_argument("--val_annotations", type = str , default = "")

    parser.add_argument("--epochs", type = int, default = 5 )
    parser.add_argument("--batch_size", type = int, default = 2 )
    parser.add_argument("--val_batch_size", type = int, default = 2 )
    parser.add_argument("--load_weights", type = str , default = "")

    parser.add_argument("--model_name", type = str , default = "vgg_segnet")
    parser.add_argument("--optimizer_name", type = str , default = "adam")

    parser.add_argument("--train_list", type = str)
    parser.add_argument("--val_list", type = str)
    parser.add_argument("--save_history_path", type = str)

    args = parser.parse_args()
    
    train_images_path = args.train_images
    train_segs_path = args.train_annotations
    train_batch_size = args.batch_size
    n_classes = args.n_classes
    input_height = args.input_height
    input_width = args.input_width
    validate = args.validate
    save_weights_path = args.save_weights_path
    epochs = args.epochs
    load_weights = args.load_weights

    optimizer_name = args.optimizer_name
    model_name = args.model_name

    if validate:
        val_images_path = args.val_images
        val_segs_path = args.val_annotations
        val_batch_size = args.val_batch_size

    modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
    modelFN = modelFns[ model_name ]
    
    is_load_weights = (len( load_weights ) > 0)

    model = modelFN( n_classes , input_height=input_height, input_width=input_width   )
    #model = deeplab.Deeplabv3(input_shape=(512, 512, 3), classes=n_classes)
    
    miou_metric = MeanIoU(n_classes)
    #adam = optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.999)
    adam = optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
          optimizer=adam ,
          metrics=['accuracy', miou_metric.mean_iou])
    
    if (is_load_weights):
        model.load_weights(load_weights)
        print("weights loaded!!!!!!!!!!!!!!!")

    print ("Model output shape" ,  model.output_shape)
    output_height = model.outputHeight
    output_width = model.outputWidth
    
    G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,
        train_batch_size,  n_classes , input_height , input_width , 
        output_height , output_width ,  list_file=args.train_list  )
    if validate:
        G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path , 
            val_batch_size,  n_classes , input_height , input_width , 
            output_height , output_width, list_file=args.val_list )
    
    with open(args.train_list) as f:
        train = f.readlines()
    train_size = len(train)
    with open(args.val_list) as f:
        val = f.readlines()
    val_size = len(val)
    print((train_size, val_size))
    steps_per_epoch = float(train_size) / args.batch_size
    validation_steps = float(val_size) / args.batch_size
    
    history = model.fit_generator( G , steps_per_epoch=steps_per_epoch , validation_data=G2 ,
            validation_steps=validation_steps ,  epochs=epochs, shuffle=False)
    model.save_weights(save_weights_path)
    model.save(save_weights_path)

    pickle.dump( history.history, open( args.save_history_path, "wb" ) )


main()




'''
if not validate:
    for ep in range( epochs ):
        model.fit_generator( G , 512  , epochs=1 )
        model.save_weights( save_weights_path + "." + str( ep ) )
        model.save( save_weights_path + ".model." + str( ep ) )
else:
    for ep in range( epochs ):
        model.fit_generator( G , args.steps_per_epoch  , validation_data=G2 , validation_steps=args.val_spe ,  epochs=1 )
        model.save_weights( save_weights_path + "." + str( ep )  )
        model.save( save_weights_path + ".model." + str( ep ) )
'''

