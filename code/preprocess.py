import gzip
import numpy as np

def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files, 
    normalizes the inputs, and returns (NumPy array of inputs, NumPy array of labels). 
    
    Read the data of the file into a buffer and use 
    np.frombuffer to turn the data into a NumPy array. Keep in mind that 
    each file has a header of a certain size. This method should be called
    within the main function of the model.py file to get BOTH the train and
    test data. 
    
    If you change this method and/or write up separate methods for 
    both train and test data, we will deduct points.
    
    :param inputs_file_path: file path for inputs, e.g. 'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, e.g. 'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather 
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs (float32) and labels (uint8)
    """
    
    # TODO: Load inputs and labels
    # TODO: Normalize inputs
    
    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        data = bytestream.read(784*num_examples)
        inputs = np.frombuffer(data,dtype=np.uint8)
        norm_input = np.divide(inputs, 255).astype(np.float32)
        input_images = np.reshape(norm_input,(-1,784))
    with open(labels_file_path, 'rb') as l, gzip.GzipFile(fileobj=l) as bytestream1:
        bytestream1.read(8)
        labels = bytestream1.read(num_examples)
        labels = np.frombuffer(labels,dtype=np.uint8)

    return input_images, labels
