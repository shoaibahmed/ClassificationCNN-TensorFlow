# ClassificationCNN-TF

Implementation of complete CNN training pipeline in TensorFlow. The data reading mechanism is based on the TF Dataset API.
<br/>The trainer currently supports six architectures including Inception ResNet v2 [1], ResNet-152 [1], NASNet [1], SE-ResNet-101 [2], SE-ResNeXt [2], and DenseNet-161 [3].

<br/>To train on a custom dataset, create a separate file for train, test and validation containing the image names and the corresponding label ids (from 0 to num_classes-1).

<br/>To initiate the training, use the command:
```
python trainer.py -t -s -v -c --batchSize 10 --trainingEpochs 10 -m NAS --displayStep 1 --weightedSoftmax
python trainer_v2.py -t -s -v -c --batchSize 10 --trainingEpochs 10 -m NAS --displayStep 1 --weightedSoftmax --trainSVM --l2Regularizer --reconstructionRegularizer
```
where -t stands for training, -s for training from scratch, -m defines the model to be used (IncResV2, ResNet, NAS), -c stands for model testing after training, and --weightedSoftmax uses weighted softmax with weights proportional to inverse frequency which helps in dealing with unbalanced classes. Training of SVM on the final feature vector is also possible by passing --trainSVM flag to trainer_v2.py along with --l2Regularizer which adds L2 regularization on the global pool feature vector and --reconstructionRegularizer which inserts a decoder network for reconstruction of the input to make sure the feature vector models all the main sources of variation in the input modality.

<br/>To initiate the testing phase, use the command:
```
python trainer.py -c -batchSize 10 -m IncResV2 --testDataFile ./data-labels-test.txt
```
where -c stands for testing and --testDataFile defines the file name to be used for testing.

Full description of flags below:

| Flag  | Description | Options   |
| ---   | ---         | ---             |
| -m    | Model architecture to use | [NAS, ResNet, IncResV2, SEResNet, SEResNeXt, DenseNet]    |
| -t    | Train model   | |
| -c    | Test model    | |
| -e    | Evaluate precision and recall per class    | |
| -s    | Start training from scratch with pre-trained weights | |
| -r    | Start training from scratch with random weights   | |
| -v    | Enable Tensorboard visualization | |
| --imageWidth  |   | |
| --imageHeight |   | |
| --imageChannels | | |
| --resizeRatio |   | |
| --useImageMean|   | |
| --learningRate| Initial learning rate  | |
| --learningRateDecay   | Enable learning rate decay | |
| --learningRateDecayTolerance | Epoch tolerance for learning rate decay | |
| --learningRateDecayFactor | Learning rate decay factor | |
| --earlyStopping | Enable early stopping | |
| --earlyStoppingTolerance | Epoch tolerance for early stopping | |
| --labelSmoothing | Enable label smoothing | |
| --weightedSoftmax | Enable softmax weighting | |
| --trainingEpochs | Number of training epochs | |
| --batchSize | Batch size | |
| --displayStep | Frequency for statistics print during training | |
| --lossFunction | Type of loss function for optimization | [cross-entropy, mse] |
| --autoAugment | Enable Tensorflow Hub's Autoaugment pre-processing| |
| --trainSVM | Train SVM on top of CNN features | |
| --reconstructionRegularizer | Enable autoencoder | |
| --reconstructionRegularizationLambda | | |
| --l2Regularizer | Enable regularization | |
| --l2RegularizationLambda | | |
| --decoderNumFilters | | |
| --modelDir | Model output directory | |
| --modelName | Model name | |
| --trainDataFile | Path to training data file | |
| --valDataFile | Path to validation data file | |
| --testDataFile | Path to testing data file | |
| --imageBaseDir | Path to image data | |
| --classesFile | Path to classes File | |

**Note:**
If classesFile is used for evaluation charts, a csv in following format should be provided: 

```
Label|Category Name
0|ClassName1
1|ClassName2
.
.
.
```

### 3rd Party Repos used:
https://github.com/chentinghao/download_google_drive \
https://github.com/tensorflow/models \
https://github.com/pudae/tensorflow-densenet \
https://github.com/HiKapok/TF-SENet \

<br/><br/> Author: <b>Shoaib Ahmed Siddiqui</b>
<br/> Email: <b>12bscsssiddiqui@seecs.edu.pk</b>

[1] https://github.com/tensorflow/models/tree/master/research/slim \
[2] https://github.com/HiKapok/TF-SENet \
[3] https://github.com/pudae/tensorflow-densenet \