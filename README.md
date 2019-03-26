# Car classification using transfer learning

In this project, I will use Convolutional Neural Network (CNN) to classify 10 different car models. Typically, to tackle this project, large datasets and domain-specific features are needed to best fit the data. However, the [dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) from stanford has limited data, each car class only has 40 images to train, and each image consists of a car in the foregound against various backgrounds and viewed fom various angles under various illuminations. Thus, the main challenge for this project is unargubly the very fine differences between different classes. Typically to learn these minute differences, a large dataset is needed. However, the hardest task for deep learning is the data itself. Thus, I decided to use out of the box deep learning frameworkd to fine-tune pre-trained classifiers for a specific fine-grained classification test. This method is known as the transfer learning. Besides, due to limited computing resouce (only one 1060 6GB GPU), I decided only train 10 car models to verify the feasibility and effectiveness of the transfer learning. 

## Introduction
I will use 3 pretrained famous classifiers as the base models of transfer learning. They are vgg19, resnet 50 and InceptionV3. The following plot is the comparison of these three models training on the [imageNet](http://www.image-net.org/) database. 

<p align="center">
  <img width="600" height="400" src="https://github.com/butroy/CNN_Car_Classification/blob/master/plots/network%20comparison.png">
</p>


For these model structures, I didn't include the fully-connected layers at the top of network. Instead, I add a global averaging pooling layer, one 512-perceptron fully connected layer, one 256-perceptron fully connected layer and one 0.3 dropout layer. I also used [Keras](https://keras.io/), a deep learning framework to construct, train, and test the networks.
 
**keras batch normalization bug:**
However, if you want to replicate my job, there is one thing you need to pay attention to: the BatchNormalization in Keras has a small bug while using transfer learning. "The problem with the current implementation of Keras is that when a BN layer is frozen, it continues to use the mini-batch statistics during training."  As a result, if you fine-tune the top layers, their weights will be adjusted to the mean/variance of the new dataset. Nevertheless, during inference they will receive data which are scaled differently because the mean/variance of the original dataset will be used. [Here](http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/) is a detailed explanation of the bug itself. To fix this bug:
```
pip install -U --force-reinstall --no-dependencies git+https://github.com/datumbox/keras@bugfix/trainable_bn
```

## Experiment
To test the power of the transfer learning, I compare the models trained under different number of frozen layers. Each framework will be either initialized with trained Imagenet weights or random weights depending on requirements. 

Below are several terms:

**Frozen:** all layers are untrianable except the custmoized layers I added myself.

**last # layers trainble:** last # of layers trainble and these layers include the custmoized layers

**train from scatch:** the whole framework is trainable and the initial weights will be randomly assigned.

## Results and discussion

| id   | Framework    | Requirement              | Validation accuracy |
|------|-------------|--------------------------|---------------------|
| id1  | vgg19       | frozen                   | 0.593052108         |
| id2  | vgg19       | last 6 layer trainable   | 0.607940447         |
| id3  | vgg19       | all layers trainable     | 0.677419356         |
| id4  | vgg19       | train from scratch       | 0.17369727          |
| id5  | resnet50    | frozen                   | **0.692307692**     |
| id6  | resnet50    | last 10 layers trainable | 0.682382134         |
| id7  | resnet50    | last 40 layers trainable | 0.625310173         |
| id8  | resnet50    | all layers trainable     | 0.64516129          |
| id9  | resnet50    | train from scratch       | 0.101736973         |
| id10 | inceptionV3 | frozen                   | 0.602977668         |
| id11 | inceptionV3 | last 10 layers trainable | 0.607940448         |
| id12 | inceptionV3 | last 40 layers trainable | 0.647642679         |
| id13 | inceptionV3 | all layers trainable     | 0.607940447         |
| id14 | inceptionV3 | train from scratch       | 0.275434243         |

Most papers about classifications use top-k accuracy as the metric to measure the performance. However, I only have 10 classes, use top-k accuracy can't demonstrate the effectiveness of the model. Thus, I only focus on right-or-wrong metric to measure.


From the table, we could observe that **initialized weights by using trained Imagenet weights is much better than using randomly initialized weights.** vgg19 and InceptionV3 perform very similar, giving validation accuracy about 60%. Resnet50 performs a little better than the other two, achieving accuracy up to 69%. Below is the individual comparison of each framework.


vgg19           |  resnet50 |inceptionV3  
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/butroy/CNN_Car_Classification/blob/master/plots/vgg19.png)  |  ![](https://github.com/butroy/CNN_Car_Classification/blob/master/plots/resnet50.png)|![](https://github.com/butroy/CNN_Car_Classification/blob/master/plots/inceptionV3.png)

And here is the confusion matrix and the training history of id5

<p align="center">
  <img width="400" height="400" src="https://github.com/butroy/CNN_Car_Classification/blob/master/plots/id5_cm.png">
   <img width="300" height="400" src="https://github.com/butroy/CNN_Car_Classification/blob/master/plots/id5_hist.png">
</p>



## Conclusion
Considering the limited source of data, this result is satisfiable and could prove that the transfer learning works well on this car classification task. In the end, I also do one full training on the whole car dataset with 196 classes, and reach the top-5 accuracy 78%. For future jobs, we can teset several more CNN models and maybe we could find some other models better fits for the car classificaiton task. 
