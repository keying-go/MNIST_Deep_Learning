from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data
        
class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with 
    batched learning. Please implement the TODOs for the entire 
    model but do not change the method and constructor arguments. 
    Make sure that your Model class works with multiple batch 
    sizes. Additionally, please exclusively use NumPy and 
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 784 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 100
        self.learning_rate = 0.5

        # TODO: Initialize weights and biases
        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros((1, self.num_classes))

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        logits = np.dot(inputs,self.W) + self.b
        e_l = np.exp(logits) # calculate the e_l for each logits
        e_sum = np.sum(e_l,axis = 1).reshape(-1, 1) # sum of e_l per image
        prob = e_l / e_sum.reshape(-1, 1)# probabilities for each class per image
        return prob
    
    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be decreasing with every training loop (step). 
        NOTE: This function is not actually used for gradient descent 
        in this assignment, but is a sanity check to make sure model 
        is learning.

        :param probabilities: matrix that contains the probabilities 
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: Calculate average cross-entropy loss for a batch
        pc = np.zeros(self.batch_size)
        # record only the prob of the correct label 
        for i in range(self.batch_size):
            pc[i] = probabilities[i,labels[i]]
        
        # cross-entropy loss for each image
        loss = -np.log(pc)
        mean_loss = np.mean(loss)
        #print(mean_loss)
        return mean_loss
    
    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases 
        after one forward pass and loss calculation. The learning 
        algorithm for updating weights and biases mentioned in 
        class works for one image, but because we are looking at 
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.

        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each 
        class for each image
        :param labels: true labels
        :return: gradient for weights, and gradient for biases
        """
        # TODO: Calculate the gradients for the weights and the gradients for the bias with respect to average loss
    
        # "one-hot encoding" for labels, the true label has a value of 1, others has a value of 0
        y = np.zeros((len(labels),self.num_classes))
        y = np.zeros((self.batch_size,self.num_classes))
        for i in range(self.batch_size):
            y[i,labels[i]] = 1
        # the gradients are the dot products of the inputs and (y_j - p_j) 
        # for batch size > 1, each entry of the dot products would be a 
        # sum of the grad of the parameter for all images in the batch, 
        # so divide by batch size to get the average gradient
        # result is a (784 x 10) matrix, each is a gradient for a weight 
        gradW = np.dot(inputs.T,(probabilities - y))/ self.batch_size
        # calculate gradient for bias
        gradB = np.sum((probabilities - y),axis = 0)/ self.batch_size
        return gradW, gradB
    
    
    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: batch accuracy (float [0,1])
        """
        # TODO: Calculate the batch accuracy
        # make the prediction and reshape it into a column
        pred = np.argmax(probabilities, axis = 1).reshape(-1,1)
        accuracy = np.sum(pred == labels) / len(labels)
        return accuracy

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient 
        descent on the Model's parameters.
        
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: Change the weights and biases of the model to descend the gradient
        
        self.W -= self.learning_rate * gradW
        self.b -= self.learning_rate * gradB
    
def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward 
    pass and backward pass
    
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments and do forward pass
    # TODO: For every batch, compute and then descend the gradients for the model's weights
    # Optional TODO: Call visualize_loss and observe the loss per batch as the model trains
    t = int(train_inputs.shape[0] / model.batch_size)
    losses = []
    for i in range(t):
        inputs = train_inputs[model.batch_size * i : model.batch_size * (i + 1), :]
        labels = train_labels[model.batch_size * i : model.batch_size * (i + 1)]
        probabilities = model.call(inputs)
        losses.append(model.loss(probabilities, labels))
        gradW, gradB = model.back_propagation(inputs, probabilities, labels)
        model.gradient_descent(gradW, gradB)
    visualize_loss(losses) # show the loss per batch as the model trains
    

    

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment, 
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy (float [0,1])
    """
    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    
    prob = model.call(test_inputs)
    accuracy = model.accuracy(prob, test_labels)
    return accuracy


def visualize_loss(losses):
    """
    NOTE: DO NOT EDIT

    Uses Matplotlib to visualize loss per batch. Call this in train().
    When you observe the plot that's displayed, think about:
    1. What does the plot demonstrate or show?
    2. How long does your model need to train to reach roughly its best accuracy so far, 
    and how do you know that?
    Optionally, add your answers to README!
    
    :param losses: an array of loss value from each batch of train
    :return: does not return anything, a plot should pop-up
    """
    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses)
    plt.show()

def visualize_results(image_inputs, probabilities, image_labels):
    """
    NOTE: DO NOT EDIT

    Uses Matplotlib to visualize the results of our model.
    
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()
    :return: does not return anything, a plot should pop-up 
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in MNIST data, initialize your model, and train and test your model 
    for one epoch. The number of training steps should be your the number of 
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    
    :return: None
    '''

    # TODO: load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    train_inputs, train_labels = get_data("../../data/train-images-idx3-ubyte.gz", "../../data/train-labels-idx1-ubyte.gz", 60000)
    test_inputs, test_labels = get_data("../../data/t10k-images-idx3-ubyte.gz", "../../data//t10k-labels-idx1-ubyte.gz", 10000)
 
    # TODO: Create Model
    model = Model()

    # TODO: Train model by calling train() ONCE on all data
    train(model, train_inputs, train_labels)

    # TODO: Test the accuracy by calling test() after running train()
    accuracy = test(model, test_inputs, test_labels)
    print('model accuracy:', accuracy)

    # TODO: Visualize the data by using visualize_results()
    image_inputs = test_inputs[0:10, :]
    probabilities = model.call(image_inputs)
    image_labels = test_labels[0:10]
    visualize_results(image_inputs, probabilities, image_labels)
    

    
if __name__ == '__main__':
    main()
