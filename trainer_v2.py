from __future__ import division
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import tensorflow_hub as hub
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import seaborn as sns
import pandas as pd

from optparse import OptionParser
from sklearn import svm

import tarfile
import time
import wget
import math
import sys
import os

import Repos.models.research.slim.nets.inception_resnet_v2 as inception_resnet_v2
import Repos.models.research.slim.nets.resnet_v1 as resnet_v1
import Repos.models.research.slim.nets.nasnet.nasnet as nasnet
import Repos.TFSENet.se_resnet as senet
import Repos.tensorflow_densenet.nets.densenet as densenet

import Repos.download_google_drive.download_gdrive as driveGet

TRAIN = 0
VAL = 1
TEST = 2

if sys.version_info[0] == 3:
    print("Using Python 3")
    import pickle as cPickle
else:
    print("Using Python 2")
    import cPickle

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-m", "--model", action="store", type="string", dest="model", default="DenseNet", help="Model to be used for Cross-Layer Pooling")
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-e", "--evalClasses", action="store_true", dest="evalClasses", default=False, help="Evaluate precision and recall per class")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch", default=False, help="Start training from scratch")
parser.add_option("-r", "--startTrainingRandomly", action="store_true", dest="startTrainingRandomly", default=False, help="Train model with randomly initialized weights")
parser.add_option("-v", "--tensorboardVisualization", action="store_true", dest="tensorboardVisualization", default=False, help="Enable tensorboard visualization")

# Input Reader Params
parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=224, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=224, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in the image")
parser.add_option("--resizeRatio", action="store", type="float", dest="resizeRatio", default=1.15, help="Resizing image ratio")
parser.add_option("--useImageMean", action="store_true", dest="useImageMean", default=False, help="Use image mean for normalization")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-4, help="Learning rate")
parser.add_option("--learningRateDecay", action="store_true", dest="learningRateDecay", default=False, help="Activate decaying learning rate")
parser.add_option("--learningRateDecayTolerance", action="store", type="int", dest="learningRateDecayTolerance", default=25,
                  help="Tolerated plateauing steps until learning rate decay")
parser.add_option("--learningRateDecayFactor", action="store", type="float", dest="learningRateDecayFactor", default=0.9, help="Factor for learning rate decay")
parser.add_option("--earlyStopping", action="store_true", dest="earlyStopping", default=False, help="Activate early stopping")
parser.add_option("--earlyStoppingTolerance", action="store", type="int", dest="earlyStoppingTolerance", default=5, help="Tolerance for early stopping")
parser.add_option("--labelSmoothing", action="store", type="float", dest="labelSmoothing", default=0.1, help="Label smoothing parameter")
parser.add_option("--weightedSoftmax", action="store_true", dest="weightedSoftmax", default=False, help="Use weighted softmax")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=1, help="Training epochs")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=5, help="Batch size")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=5, help="Progress display step")
parser.add_option("--lossFunction", action="store", type="string", dest="lossFunction", default="cross_entropy", help="Loss function to use for training")
parser.add_option("--autoAugment", action="store_true", dest="autoAugment", default=False, help="Use google's Autoaugment pre-processing")

parser.add_option("--trainSVM", action="store_true", dest="trainSVM", default=False, help="Train SVM on top of the features extracted from the trained model")
parser.add_option("--reconstructionRegularizer", action="store_true", dest="reconstructionRegularizer", default=False,
                  help="Add decoder at the end for reconstruction of the original input signal")
parser.add_option("--reconstructionRegularizationLambda", action="store", type="float", dest="reconstructionRegularizationLambda", default=1e-4,
                  help="Reconstruction regularization parameter")
parser.add_option("--l2Regularizer", action="store_true", dest="l2Regularizer", default=False, help="Add L2 regularization on the final feature vector obtained after global pool")
parser.add_option("--l2RegularizationLambda", action="store", type="float", dest="l2RegularizationLambda", default=1e-4, help="L2 regularization parameter")
parser.add_option("--decoderNumFilters", action="store", type="int", dest="decoderNumFilters", default=64, help="Number of filters in the decoder layers")

# Directories
parser.add_option("--modelDir", action="store", type="string", dest="modelDir", default="./Models/mymodel/", help="Directory for saving the model")
parser.add_option("--modelName", action="store", type="string", dest="modelName", default="mymodel", help="Name to be used for saving the model")

parser.add_option("--trainDataFile", action="store", type="string", dest="trainDataFile", default="Data/small-title28cat-labels-test.txt",
                  help="Training data file")
parser.add_option("--valDataFile", action="store", type="string", dest="valDataFile", default="Data/small-title28cat-labels-test.txt",
                  help="Validation data file")
parser.add_option("--testDataFile", action="store", type="string", dest="testDataFile", default="Data/small-title28cat-labels-test.txt",
                  help="Test data file")

parser.add_option("--imageBaseDir", action="store", type="string", dest="imageBaseDir", default="/home/adri/Data/title30cat/56x56/", help="Image base directory")
parser.add_option("--classesFile", action="store", type="string", dest="classesFile", default=None, help="Path to classes file")

# Parse command line options
(options, args) = parser.parse_args()
print(options)

baseDir = os.getcwd()
usingGlobalPooledFeatures = False

if options.autoAugment:
    # Initialize autoaugment module
    augmentation_module = hub.Module('https://tfhub.dev/google/image_augmentation/crop_color/1')

# Create folder for pre-trained weights
if not os.path.exists('Pre-trained/'):
    os.makedirs('Pre-trained/')

# Load the pre-trained model
if options.model == "ResNet":
    resnet_checkpoint_file = os.path.join(baseDir, 'Pre-trained/resnet/resnet_v1_152.ckpt')
    if not os.path.isfile(resnet_checkpoint_file):
        # Download file from the link
        url = 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'
        filename = wget.download(url, './Pre-trained')
        
        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/resnet/')
        tar.close()
        
        os.remove(filename)
    
    options.imageHeight = options.imageWidth = 224

elif options.model == "IncResV2":
    inc_res_v2_checkpoint_file = os.path.join(baseDir, 'Pre-trained/incresv2/inception_resnet_v2_2016_08_30.ckpt')
    if not os.path.isfile(inc_res_v2_checkpoint_file):
        # Download file from the link
        url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
        filename = wget.download(url, './Pre-trained')
        
        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/incresv2/')
        tar.close()
        
        os.remove(filename)
    
    options.imageHeight = options.imageWidth = 299

elif options.model == "NAS":
    nas_checkpoint_file = os.path.join(baseDir, 'Pre-trained/NAS/model.ckpt')
    if not os.path.isfile(nas_checkpoint_file + '.index'):
        # Download file from the link
        url = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
        filename = wget.download(url, './Pre-trained')
        
        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/NAS/')
        tar.close()
        
        os.remove(filename)
    
    # Update image sizes
    options.imageHeight = options.imageWidth = 331

elif options.model == "DenseNet":
    densenet_checkpoint_file = os.path.join(baseDir, 'Pre-trained/tf-densenet161/tf-densenet161.ckpt')
    if not os.path.isfile(densenet_checkpoint_file + '.index'):
        # Download file from the link
        id = '0B_fUSpodN0t0NmZvTnZZa2plaHc'
        filename = './Pre-trained/tf-densenet161.tar.gz'
        driveGet.download_file_from_google_drive(id, filename)
        
        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/tf-densenet161/')
        tar.close()
        
        os.remove(filename)
    
    # Update image sizes
    options.imageHeight = options.imageWidth = 224

elif options.model == "SENet":
    senet_checkpoint_file = os.path.join(baseDir, 'Pre-trained/seresnet101/se_resnet101.ckpt')
    if not os.path.isfile(senet_checkpoint_file + '.index'):
        # Download file from the link
        id = '19QsGHNZC0BVsaDf4Sx79J2Hl2BV9wpRm'
        filename = './Pre-trained/seresnet101.tar.gz'
        driveGet.download_file_from_google_drive(id, filename)
        
        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/seresnet101/')
        tar.close()
        
        os.remove(filename)
    
    # Update image sizes
    options.imageHeight = options.imageWidth = 224

else:
    print("Error: Unknown model selected")
    exit(-1)

# Define params
IMAGENET_MEAN = [123.68, 116.779, 103.939]  # RGB

# Decide the resizing dimensions
RESIZED_IMAGE_DIMS = [int(options.imageHeight * options.resizeRatio), int(options.imageWidth * options.resizeRatio)]
print("Resized image dimensions: %s" % str(RESIZED_IMAGE_DIMS))


# Reads an image from a file, decodes it into a dense tensor
def _parse_function_train(filename, label):
    image_string = tf.read_file(filename)
    
    if not options.autoAugment:
        img = tf.image.decode_jpeg(image_string)
    
        img = tf.image.resize_images(img, RESIZED_IMAGE_DIMS)
    
        # Random crop
        img = tf.random_crop(img, [options.imageHeight, options.imageWidth, options.imageChannels])
    
        # Random flipping
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    
        img.set_shape([options.imageHeight, options.imageWidth, options.imageChannels])
        img = tf.cast(img, tf.float32)  # Convert to float tensor
        return img, filename, label
    else:
        return image_string, filename, label


# Reads an image from a file, decodes it into a dense tensor
def _parse_function_test(filename, label):
    image_string = tf.read_file(filename)
    if not options.autoAugment:
        img = tf.image.decode_jpeg(image_string)

        img = tf.image.resize_images(img, [options.imageHeight, options.imageWidth])
    
        img.set_shape([options.imageHeight, options.imageWidth, options.imageChannels])
        img = tf.cast(img, tf.float32)  # Convert to float tensor
        return img, filename, label
    else:
        return image_string, filename, label


def loadDataset(currentDataFile, mode):
    print("Loading data from file: %s" % (currentDataFile))
    dataClasses = {}
    with open(currentDataFile) as f:
        imageFileNames = f.readlines()
        imNames = []
        imLabels = []
        for imName in imageFileNames:
            imName = imName.strip().split('|')
            imNames.append(os.path.join(options.imageBaseDir, imName[0]))
            currentLabel = int(imName[1])
            imLabels.append(currentLabel)

            if currentLabel not in dataClasses:
                dataClasses[currentLabel] = 1
            else:
                dataClasses[currentLabel] += 1

        imNames = tf.constant(imNames)
        imLabels = tf.constant(imLabels)

    numClasses = len(dataClasses)
    numFiles = len(imageFileNames)
    print("Dataset loaded")
    print("Files: %d | Classes: %d" % (numFiles, numClasses))
    print(dataClasses)
    classWeights = [float(numFiles - dataClasses[x]) / float(numFiles) for x in dataClasses]
    print("Class weights: %s" % str(classWeights))

    dataset = tf.data.Dataset.from_tensor_slices((imNames, imLabels))

    # Select pre-processing pipeline
    if mode == TRAIN:
        dataset = dataset.map(_parse_function_train)
    elif mode == VAL or mode == TEST:
        dataset = dataset.map(_parse_function_test)
    else:
        print("Error: Unknown argument for pre-processing")
        exit(-1)

    dataset = dataset.shuffle(buffer_size=numFiles)
    dataset = dataset.batch(options.batchSize)
    dataset = dataset.prefetch(buffer_size=numFiles)

    return dataset, numClasses, classWeights

# A vector of filenames
trainDataset, numClasses, classWeights = loadDataset(options.trainDataFile, TRAIN)
valDataset, _, _ = loadDataset(options.valDataFile, VAL)
testDataset, _, _ = loadDataset(options.testDataFile, TEST)

trainIterator = trainDataset.make_initializable_iterator()
valIterator = valDataset.make_initializable_iterator()
testIterator = testDataset.make_initializable_iterator()

global_step = tf.train.get_or_create_global_step()


def getShapeAsList(tensor):
    tensorShape = tensor.get_shape()
    shapeList = [-1]
    for val in tensorShape[1:]:
        shapeList.append(int(val))
    return shapeList


def lrelu_func(x, leak=0.2):
    return tf.maximum(x, leak * x)


def getDeconvFilter(f_shape):
    width = f_shape[0]
    height = f_shape[1]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
    
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


def upscoreLayer(bottom, shape, num_outputs, name,
                 num_in_features, activation=lrelu_func,
                 ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        # in_features = bottom.get_shape()[3].value
        in_features = num_in_features
        
        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)
            # h = ((in_shape[1] - 1) * stride) + 1
            # w = ((in_shape[2] - 1) * stride) + 1
            h = in_shape[1] * stride
            w = in_shape[2] * stride
            new_shape = [in_shape[0], h, w, num_outputs]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_outputs]
        output_shape = tf.stack(new_shape)
        
        f_shape = [ksize, ksize, num_outputs, in_features]
        
        # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input) ** 0.5
        
        weights = getDeconvFilter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')
        
        # Set the shape to make it more specific
        deconv.set_shape([None, int(bottom.get_shape()[1] * stride), int(bottom.get_shape()[2] * stride), num_outputs])
        
        if activation is not None:
            deconv = activation(deconv)
    
    return deconv


def decoderModule(featureVec, requiredShape):
    # Keep adding layers until the output is equal or exceeds the required output size
    # Reshape the input
    if usingGlobalPooledFeatures:
        output = tf.reshape(featureVec, shape=[-1, 1, 1, int(featureVec.get_shape()[1])])
    else:
        output = featureVec
    currentDataShape = getShapeAsList(output)
    layerIdx = 1
    convTransposeStride = 2
    print("Feature vector shape: %s | Decoder input shape: %s | Required shape: %s" % (featureVec.get_shape(), output.get_shape(), requiredShape))
    
    while True:
        currentDataShape = getShapeAsList(output)
        
        # Check if the size exceeds or is equal
        if (currentDataShape[1] >= requiredShape[1]) and (currentDataShape[2] >= requiredShape[2]):
            if (currentDataShape[1] > requiredShape[1]) and (currentDataShape[2] > requiredShape[2]):
                # Get the cenral crop
                startingOffset = [int((output.get_shape()[1] - requiredShape[1]) / 2), int((output.get_shape()[2] - requiredShape[2]) / 2)]
                output = output[:, startingOffset[0]: startingOffset[0] + requiredShape[1], startingOffset[1]: startingOffset[1] + requiredShape[2], :]
                print("Decoder cropped output shape: %s" % (output.get_shape()))
            
            # Add the final convolutional layer to match the output channels
            output = tf.layers.conv2d(inputs=output, filters=requiredShape[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None, name='Decoder_Logits')
            print("Decoder Logits shape: %s" % (output.get_shape()))
            
            break
        else:
            # Add the deconvolutional layer
            output = upscoreLayer(output, shape=None, num_outputs=options.decoderNumFilters, num_in_features=currentDataShape[3], name='Upscore_' + str(layerIdx),
                                  stride=convTransposeStride)
            print("Upscore-%d layer output shape: %s" % (layerIdx, output.get_shape()))
        
        layerIdx += 1
    
    # Assertion for output size
    outputShape = output.get_shape()
    assert ((outputShape[1] == requiredShape[1]) and (outputShape[2] == requiredShape[2]) and (outputShape[3] == requiredShape[3]))
    return output


with tf.name_scope('Model'):
    # Data placeholders
    datasetSelectionPlaceholder = tf.placeholder(dtype=tf.int32, shape=(), name='DatasetSelectionPlaceholder')
    inputBatchImages, inputBatchImageNames, inputBatchLabels = tf.cond(tf.equal(datasetSelectionPlaceholder, TRAIN),
                                                                       lambda: trainIterator.get_next(),
                                                                       lambda: tf.cond(tf.equal(datasetSelectionPlaceholder, VAL),
                                                                                       lambda: valIterator.get_next(),
                                                                                       lambda: testIterator.get_next()))
    inputBatchImageLabels = tf.one_hot(inputBatchLabels, depth=numClasses)
    
    if options.autoAugment:
        # Auto-Augment only training data
        inputBatchImages = augmentation_module({
            'encoded_images': inputBatchImages,
            'image_size': [options.imageHeight, options.imageWidth],
            'augmentation': tf.equal(datasetSelectionPlaceholder, TRAIN)
        })
        inputBatchImages = tf.image.resize_images(inputBatchImages, [options.imageHeight, options.imageWidth])

    print("Data shape: %s" % str(inputBatchImages.get_shape()))
    print("Labels shape: %s" % str(inputBatchImageLabels.get_shape()))

if options.model == "IncResV2":
    with tf.name_scope('Model'):
        scaledInputBatchImages = tf.scalar_mul((1.0 / 255.0), inputBatchImages)
        scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
        scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

        # Create model
        arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            # logits, end_points = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages, is_training=True, dropout_keep_prob=0.5 if options.trainModel else 1.0, num_classes=numClasses)
            logits, end_points = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages, is_training=options.trainModel,
                                                                         num_classes=numClasses)

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        variables_to_restore = slim.get_variables_to_restore(
            exclude=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"])

        # Last layer for extraction of features before global pool
        featureVector = end_points['Mixed_7a']

elif options.model == "ResNet":
    with tf.name_scope('Model'):
        if options.useImageMean:
            imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keepdims=True)
            print("Image mean shape: %s" % str(imageMean.shape))
            processedInputBatchImages = inputBatchImages - imageMean
        else:
            print(inputBatchImages.shape)
            channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
            for i in range(options.imageChannels):
                channels[i] -= IMAGENET_MEAN[i]
            processedInputBatchImages = tf.concat(axis=3, values=channels)
            print(processedInputBatchImages.shape)

        # Create model
        arg_scope = resnet_v1.resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            # logits, end_points = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=options.trainModel, num_classes=numClasses)
            logits, end_points = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=options.trainModel,
                                                         num_classes=numClasses)

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        variables_to_restore = slim.get_variables_to_restore(
            exclude=["resnet_v1_152/logits", "resnet_v1_152/AuxLogits"])

        # Last layer for extraction of features before global pool
        featureVector = end_points['resnet_v1_152/block4']

elif options.model == "NAS":
    with tf.name_scope('Model'):
        scaledInputBatchImages = tf.scalar_mul((1.0 / 255.0), inputBatchImages)
        scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
        scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

        # Create model
        arg_scope = nasnet.nasnet_large_arg_scope()
        with slim.arg_scope(arg_scope):
            # logits, end_points = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=options.trainModel, num_classes=numClasses)
            logits, end_points = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=options.trainModel,
                                                           num_classes=numClasses)

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        variables_to_restore = slim.get_variables_to_restore(exclude=["aux_11/aux_logits/FC", "final_layer/FC"])

        # Last layer for extraction of features before global pool
        featureVector = end_points['Cell_17']

elif options.model == "DenseNet":
    if options.useImageMean:
        imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keepdims=True)
        print("Image mean shape: %s" % str(imageMean.shape))
        processedInputBatchImages = inputBatchImages - imageMean
    else:
        print(inputBatchImages.shape)
        channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
        for i in range(options.imageChannels):
            channels[i] -= IMAGENET_MEAN[i]
        processedInputBatchImages = tf.concat(axis=3, values=channels)
        print(processedInputBatchImages.shape)

    arg_scope = densenet.densenet_arg_scope()
    with slim.arg_scope(arg_scope):
        net, end_points = densenet.densenet161(processedInputBatchImages, num_classes=numClasses, is_training=options.trainModel)
        logits = tf.layers.flatten(net)

    # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
    variables_to_restore = slim.get_variables_to_restore(exclude=["densenet161/logits"])

    # Last layer for extraction of features before global pool
    featureVector = tf.get_default_graph().get_tensor_by_name("densenet161/dense_block4/conv_block24/concat:0")

elif options.model == "SENet":
    if options.useImageMean:
        imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keepdims=True)
        print("Image mean shape: %s" % str(imageMean.shape))
        processedInputBatchImages = inputBatchImages - imageMean
    else:
        print(inputBatchImages.shape)
        channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
        for i in range(options.imageChannels):
            channels[i] -= IMAGENET_MEAN[i]
        processedInputBatchImages = tf.concat(axis=3, values=channels)
        print(processedInputBatchImages.shape)

    # Convert RGB to BGR for SENet input
    processedInputBatchImages = tf.reverse(processedInputBatchImages, [-1])

    # Create model
    logits, _ = senet.SE_ResNet(processedInputBatchImages, numClasses, is_training=options.trainModel,
                                data_format='channels_last')

    # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
    variables_to_restore = slim.get_variables_to_restore(exclude=["dense", "global_step"])

    # Last layer for extraction of features
    featureVector = tf.get_default_graph().get_tensor_by_name("conv5_3/relu:0")

else:
    print("Error: Unknown model selected")
    exit(-1)
    
print("Feature Vector Dimensions: %s" % str(featureVector.get_shape()))
# Add the decoder
if options.reconstructionRegularizer:
    with tf.name_scope('Decoder'):
        print("Adding the Decoder network")
        reconstructedInput = decoderModule(featureVector, getShapeAsList(inputBatchImages))
        
        if options.tensorboardVisualization:
            tf.summary.image('Original Image', inputBatchImages, max_outputs=3)
            tf.summary.image('Reconstructed Image', reconstructedInput, max_outputs=3)

with tf.name_scope('Loss'):
    if options.weightedSoftmax:
        print("Using weighted cross-entropy loss")
        # Define the class weightages (weighted softmax)
        classWeightsTensor = tf.constant(classWeights)
        classWeights = tf.gather(classWeightsTensor, inputBatchLabels)
    else:
        print("Using unweighted cross-entropy loss")
        classWeights = tf.ones_like(inputBatchLabels)
    
    # Define loss
    if options.lossFunction == 'cross_entropy':
        main_train_loss = tf.losses.softmax_cross_entropy(onehot_labels=inputBatchImageLabels, logits=logits, weights=classWeights, label_smoothing=options.labelSmoothing)
    elif options.lossFunction == 'mse':
        main_train_loss = tf.losses.mean_squared_error(labels=inputBatchImageLabels, predictions=logits)
    else:
        print("Error: Unknown loss function")
        exit(-1)

    tf.losses.add_loss(main_train_loss)
    
    # Add L2 loss on the feature vector
    if options.l2Regularizer:
        l2_reg = options.l2RegularizationLambda * tf.reduce_sum(end_points['global_pool'])
        tf.losses.add_loss(l2_reg)
    
    # Add decoder loss
    if options.reconstructionRegularizer:
        reconstruction_loss = options.reconstructionRegularizationLambda * tf.reduce_mean(tf.square(reconstructedInput - inputBatchImages))
        tf.losses.add_loss(reconstruction_loss)
    
    loss = tf.reduce_sum(tf.losses.get_losses())

with tf.name_scope('Accuracy'):
    correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(inputBatchImageLabels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

with tf.name_scope('Optimizer'):
    # Define learning placeholder
    learningRate = tf.placeholder(dtype=tf.float32, shape=(), name='LearningRate')
    # Define Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    # if options.reconstructionRegularizer:
    # 	autoEncoderTrainOp = tf.train.AdamOptimizer(learning_rate=options.learningRate).minimize(reconstruction_loss)
    
    # Op to calculate every variable gradient
    gradients = tf.gradients(loss, tf.trainable_variables())
    gradients = list(zip(gradients, tf.trainable_variables()))
    
    # Op to update all variables according to their gradient
    # Update operation for batchnorm
    updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(updateOps):
        trainOp = optimizer.apply_gradients(grads_and_vars=gradients)

# Initializing the variables
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

if options.tensorboardVisualization:
    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    tf.summary.scalar(options.lossFunction, main_train_loss)
    tf.summary.scalar("accuracy", accuracy)
    if options.l2Regularizer:
        tf.summary.scalar("l2 regularization loss", l2_reg)
    if options.reconstructionRegularizer:
        tf.summary.scalar("reconstruction loss", reconstruction_loss)
    
    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    
    # Summarize all gradients
    for grad, var in gradients:
        if grad is not None:
            tf.summary.histogram(var.name + '/gradient', grad)
    
    # Merge all summaries into a single op
    mergedSummaryOp = tf.summary.merge_all()

# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Train model
if options.trainModel:
    with tf.Session(config=config) as sess:
        # Initialize all vars
        sess.run(init)
        sess.run(init_local)

        # Restore the model params
        if options.startTrainingFromScratch:
            # Restore pre-trained weights
            print("Removing previous checkpoints and logs")
            os.system("rm -rf " + options.modelDir + "logs")
            os.system("rm -rf " + options.modelDir)
            os.system("mkdir " + options.modelDir)

            checkpointFileName = resnet_checkpoint_file if options.model == "ResNet" else senet_checkpoint_file if options.model == "SENet" else densenet_checkpoint_file if options.model == "DenseNet" else inc_res_v2_checkpoint_file if options.model == "IncResV2" else nas_checkpoint_file
            print("Restoring weights from file: %s" % (checkpointFileName))

            # Load the imagenet pre-trained model
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, checkpointFileName)
        elif options.startTrainingRandomly:
            # Train model from scratch, without pre-trained weights, but with random initialization
            pass
        else:
            # Load the user trained model
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(options.modelDir, options.modelName))

        # Saver op to save and restore all the variables
        saver = tf.train.Saver()

        if options.tensorboardVisualization:
            # Write the graph to file
            summaryWriter = tf.summary.FileWriter(os.path.join(options.modelDir, "logs"), graph=tf.get_default_graph())

        globalStep = 0
        numEpochs = options.trainingEpochs + 1 if options.trainSVM else options.trainingEpochs
        if options.trainSVM:
            imageNames = []
            imageLabels = []
            imageFeatures = []

        # Initializing early stopping & lr decay
        earlyStoppingCounter = 0
        lrDecayCounter = 0
        bestLoss = sys.float_info.max

        # Set learning rate
        currentLearningRate = options.learningRate

        for epoch in range(numEpochs):
            # Initialize the dataset iterator
            sess.run(trainIterator.initializer)
            isLastEpoch = epoch == options.trainingEpochs
            try:
                step = 0
                while True:
                    start_time = time.time()

                    if isLastEpoch:
                        # Collect features for SVM
                        [imageName, imageLabel, featureVec] = sess.run([inputBatchImageNames, inputBatchLabels, featureVector], feed_dict={datasetSelectionPlaceholder: TRAIN})
                        imageNames.extend(imageName)
                        imageLabels.extend(imageLabel)
                        imageFeatures.extend(np.reshape(featureVec, [featureVec.shape[0], -1]))

                        duration = time.time() - start_time

                        # Print an overview fairly often.
                        if step % options.displayStep == 0:
                            print('Step: %d | Duration: %f' % (step, duration))
                    else:
                        # Run optimization op (backprop)
                        if options.tensorboardVisualization:
                            [trainLoss, currentAcc, _, summary] = sess.run([loss, accuracy, trainOp, mergedSummaryOp], feed_dict={datasetSelectionPlaceholder: TRAIN, learningRate: currentLearningRate})
                            # [trainLoss, currentAcc, _, summary] = sess.run([reconstruction_loss, accuracy, autoEncoderTrainOp, mergedSummaryOp], feed_dict={datasetSelectionPlaceholder: TRAIN})
                            summaryWriter.add_summary(summary, globalStep)
                        else:
                            [trainLoss, currentAcc, _] = sess.run([loss, accuracy, trainOp], feed_dict={datasetSelectionPlaceholder: TRAIN, learningRate: currentLearningRate})

                        duration = time.time() - start_time

                        # Print an overview fairly often.
                        if step % options.displayStep == 0:
                            print('Step: %d | Loss: %f | Accuracy: %f | Duration: %f' % (step, trainLoss, currentAcc, duration))

                    step += 1
                    globalStep += 1

            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (epoch, step))

                # Validation initialization
                sess.run(valIterator.initializer)

                # Evaluate validation data
                try:
                    val_step = 0
                    correctInstances = 0
                    totalInstances = 0
                    totalValLoss = 0.0

                    while True:
                        batchLabelsTest, predictions, valLoss = sess.run([inputBatchImageLabels, logits, loss],
                                                                         feed_dict={datasetSelectionPlaceholder: VAL})

                        predClass = np.argmax(predictions, axis=1)
                        actualClass = np.argmax(batchLabelsTest, axis=1)

                        correctInstances += np.sum(predClass == actualClass)
                        totalInstances += predClass.shape[0]
                        totalValLoss += valLoss

                        val_step += 1

                except tf.errors.OutOfRangeError:
                    valAcc = (float(correctInstances) / float(totalInstances)) * 100
                    valLoss = totalValLoss / float(val_step)
                    print('Done validating for %d steps. Validation accuracy: %0.2f%%. Valdiation loss: %0.5f' % (
                    val_step, valAcc, valLoss))

                    # Early stopping & lr decay
                    if valLoss < bestLoss:
                        
                        # Save best model only when early stopping active
                        if options.earlyStopping:
                            saver.save(sess, os.path.join(options.modelDir, 'best/' + options.modelName))
                            print("New best model saved: %s" % (os.path.join(options.modelDir, 'best/' + options.modelName)))

                        # Update variables
                        bestLoss = valLoss
                        earlyStoppingCounter = 0
                        lrDecayCounter = 0
                    else:
                        earlyStoppingCounter += 1
                        lrDecayCounter += 1
                        # Check if tolerance limit reached
                        if earlyStoppingCounter == options.earlyStoppingTolerance and options.earlyStopping:
                            break
                        # Check if lr needs to be decayed
                        if lrDecayCounter == options.learningRateDecayTolerance and options.learningRateDecay:
                            currentLearningRate *= options.learningRateDecayFactor
                            lrDecayCounter = 0
                    print("Early stopping: %d. LRDecay: %d" % (earlyStoppingCounter, lrDecayCounter))

        # Save final model weights to disk
        saver.save(sess, os.path.join(options.modelDir, 'final/' + options.modelName))
        print("Final model saved: %s" % (os.path.join(options.modelDir, 'final/' + options.modelName)))

        if options.trainSVM:
            # Train the SVM
            print("Training SVM")
            imageFeatures = np.array(imageFeatures)
            imageLabels = np.array(imageLabels)
            print("Data shape: %s" % str(imageFeatures.shape))
            print("Labels shape: %s" % str(imageLabels.shape))

            clf = svm.LinearSVC(C=1.0)
            clf.fit(imageFeatures, imageLabels)
            print("Training Complete!")

            with open(os.path.join(options.modelDir, 'svm.pkl'), 'wb') as fid:
                cPickle.dump(clf, fid)

            print("Evaluating performance on training data")
            trainAccuracy = clf.score(imageFeatures, imageLabels)
            print("Train accuracy: %f" % (trainAccuracy))

    print("Optimization Finished!")

# Test model
if options.testModel:
    print("Testing saved model")

    if options.evalClasses:
        conf_matrix = np.array(np.zeros((numClasses, numClasses)))
        precision = np.array(np.zeros((numClasses)))
        recall = np.array(np.zeros((numClasses)))
        total_prec = 0
        total_rec = 0

    # Now we make sure the variable is now a constant, and that the graph still produces the expected result.
    with tf.Session(config=config) as sess:
        # Saver op to save and restore all the variables
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(options.modelDir, options.modelName))

        # Initialize the dataset iterator
        sess.run(testIterator.initializer)

        svmFound = False
        if os.path.exists(os.path.join(options.modelDir, 'svm.pkl')):
            print("Loading saved SVM instance")
            with open(os.path.join(options.modelDir, 'svm.pkl'), 'rb') as fid:
                clf = cPickle.load(fid)
                if clf is None:
                    print("Error: Unable to load SVM instance.")
                    exit(-1)
                svmFound = True
                print("SVM instance loaded successfully!")

        try:
            step = 0
            correctInstances = 0
            totalInstances = 0

            if svmFound:
                correctInstancesSVM = 0
                imageLabels = []
                imageFeatures = []

            while True:
                start_time = time.time()

                [batchLabelsTest, predictions, currentAcc, featureVec] = sess.run([inputBatchImageLabels, logits, accuracy, featureVector], feed_dict={datasetSelectionPlaceholder: TEST})

                predConf = np.max(predictions, axis=1)
                predClass = np.argmax(predictions, axis=1)
                actualClass = np.argmax(batchLabelsTest, axis=1)

                # Create confusion matrix
                if options.evalClasses:
                    mask = np.equal(predClass, actualClass)
                    for idx, el in enumerate(mask):
                        if el:
                            conf_matrix[actualClass[idx], actualClass[idx]] += 1
                        else:
                            conf_matrix[predClass[idx], actualClass[idx]] += 1

                correctInstances += np.sum(predClass == actualClass)
                totalInstances += predClass.shape[0]

                if svmFound:
                    imageLabels.extend(actualClass)
                    imageFeatures.extend(np.reshape(featureVec, [featureVec.shape[0], -1]))

                duration = time.time() - start_time
                print('Step: %d | Accuracy: %f | Duration: %f' % (step, currentAcc, duration))

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done testing for %d epochs, %d steps.' % (1, step))

    print('Number of test images: %d' % (totalInstances))
    print('Number of correctly predicted images: %d' % (correctInstances))
    print('Test set accuracy: %f' % ((float(correctInstances) / float(totalInstances)) * 100))

    if options.evalClasses:
        # Calculate precision and recall for each class
        for i in range(numClasses):
            precision[i] = np.divide(conf_matrix[i, i], np.sum(conf_matrix[i, :]), out=np.zeros_like(conf_matrix[i, i]),
                                     where=conf_matrix[i, i] != 0)
            recall[i] = np.divide(conf_matrix[i, i], np.sum(conf_matrix[:, i]), out=np.zeros_like(conf_matrix[i, i]),
                                  where=conf_matrix[i, i] != 0)

        # Calculate overall precision and recall
        total_prec = np.mean(precision)
        total_rec = np.mean(recall)

        # Map class names if classesFile is given
        if options.classesFile is not None:
            df_classes = pd.read_csv(options.classesFile, sep='|')
        else:
            df_classes = pd.DataFrame(np.arange(numClasses), columns=['Category Name'])
            df_classes = "Class " + df_classes

        # Plot confusion matrix
        fig_cm = plt.figure(figsize=(11, 11))
        ax_cm = fig_cm.add_subplot(111)
        sns.set(font_scale=0.7)  # for label size
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="inferno")
        ax_cm.set_xticklabels(list(df_classes['Category Name']), rotation=90)
        ax_cm.set_yticklabels(list(df_classes['Category Name']), rotation=0)
        ax_cm.set_title('Testing confusion matrix (%d test images)' % totalInstances)
        fig_cm.tight_layout()
        fig_cm.savefig(os.path.join(options.modelDir, '01_Test_confusion-matrix.jpg'), dpi=500)

        # Plot recall and precision per class
        ind = np.arange(numClasses)  # the x locations for the groups
        width = 0.35  # the width of the bars
        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111)
        rects1 = ax.bar(ind, precision, width, color='royalblue')
        rects2 = ax.bar(ind + width, recall, width, color='seagreen')
        ax.set_ylabel('Scores')
        ax.set_title(
            'Scores by class (Total Precision: %.2f%%, Total Recall: %.2f%%' % (total_prec * 100, total_rec * 100))
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(list(df_classes['Category Name']))
        ax.legend((rects1[0], rects2[0]), ('Precision', 'Recall'))
        plt.xticks(rotation=90)
        fig.tight_layout()
        fig.savefig(os.path.join(options.modelDir, '02_Metrics_per_class.jpg'), dpi=500)

    print('Number of test images: %d' % (totalInstances))
    print('Number of correctly predicted images: %d' % (correctInstances))
    print('Test set accuracy: %f' % ((float(correctInstances) / float(totalInstances)) * 100))

    if svmFound:
        print("Evaluating SVM")
        imageFeatures = np.array(imageFeatures)
        imageLabels = np.array(imageLabels)
        print("Data shape: %s" % str(imageFeatures.shape))
        print("Labels shape: %s" % str(imageLabels.shape))

        testAccuracy = clf.score(imageFeatures, imageLabels)
        print("Test accuracy: %f" % (testAccuracy))