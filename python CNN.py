import numpy as np
import cupy as cp
from cupy.lib.stride_tricks import as_strided
from mnist import MNIST
import time
import json
import math
import cupyx
from gpu_conv import conv2d,max_pool
# Load MNIST dataset
mndata = MNIST("MNIST_ORG")
training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Convert to numpy arrays and normalize
training_images = cp.array(training_images) / 255.0
test_images = cp.array(test_images) / 255.0
training_labels=cp.array(training_labels)
test_labels=cp.array(test_labels)

def to_gpu(data):
    if isinstance(data, np.ndarray):
        return cp.asarray(data)
    return data

def to_cpu(data):
    if isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return data

class PoolLayer:
    def __init__(self, input_shape, pool_size=(2, 2), stride=2, pool_type="max",batch_size=100):
        """
        Initialize a pooling layer optimized for GPU execution using CuPy with batch processing.
        
        Args:
            input_shape: Tuple of (batch_size, num_channels, input_width, input_height)
            pool_size: Tuple of (pool_width, pool_height)
            stride: Stride size for pooling operation
            pool_type: Type of pooling operation ('max' or 'avg')
        """
        batch_size,num_channels, input_width, input_height = input_shape
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.stride = stride
        
        # Calculate output dimensions
        pool_width, pool_height = pool_size
        output_width = math.floor((input_width - pool_width) / stride + 1)
        output_height = math.floor((input_height - pool_height) / stride + 1)
        self.output_shape = (batch_size, num_channels, output_width, output_height)
        
        # Initialize arrays
        self.input_images = None
        self.output_images = None
        self.max_indices = None  # Store indices for max pooling backprop
        self.input_derivatives = None
    
    
    def forward_pass(self, input_images):
        """
        Forward pass using vectorized operations for GPU efficiency with batch processing.
        
        Args:
            input_images: Input tensor of shape (batch_size, num_channels, input_width, input_height)
            
        Returns:
            Output tensor after pooling
        """
        batch_size,num_channels, input_width, input_height = self.input_shape
        output_width, output_height = self.output_shape[2:]
        shape=(batch_size,num_channels,output_width,output_height,self.pool_size[0],self.pool_size[1])
        strides =(input_images.strides[0],input_images.strides[1],input_images.strides[2]*self.stride,input_images.strides[3]*self.stride,input_images.strides[2],input_images.strides[3])
        windows = as_strided(input_images,shape=shape,strides=strides).reshape(batch_size,num_channels,output_width,output_height,-1)
        self.max_indices=cp.argmax(windows,axis=4)
        self.output_images=cp.max(windows,axis=4)
    
    def backward_pass(self, gradient_images):
        """
        Fully vectorized backward pass for pooling with batch processing.
        
        Args:
            gradient_images: Gradient tensor of shape (batch_size, num_channels, output_width, output_height)
            
        Returns:
            Gradient with respect to input
        """
        batch_size,num_channels, input_width, input_height = self.input_shape
        pool_width, pool_height = self.pool_size
        output_width, output_height = self.output_shape[2:]
        
        # Initialize input derivatives
        self.input_derivatives = cp.zeros(self.input_shape, dtype=gradient_images.dtype)

    
    def apply_gradients(self, learn_rate):
        """
        Pooling layers don't have parameters to update.
        """
        return
class ConvLayer:
    def __init__(self, input_shape, kernal_depth, kernal_size=(3,3)):
        batch_size,num_channels, input_width, input_height = input_shape
        
        # Don't pre-allocate large arrays
        self.input_images = None
        self.input_derivatives = None
        self.output_images = None
        
        # Store shapes for later use
        self.input_shape = input_shape
        self.output_shape = (batch_size, kernal_depth, 
                           input_width - kernal_size[0] + 1,
                           input_height - kernal_size[1] + 1)
        
        self.kernal_depth = kernal_depth
        self.num_channels = num_channels
        self.kernal_size = kernal_size
        
        # Only allocate necessary arrays
        self.biases = cp.zeros((kernal_depth, 1, 1))
        self.bias_gradients = None
        self.kernals = self.he_initialization(kernal_depth, num_channels, kernal_size)
        self.kernal_gradients = None

    def he_initialization(self, kernal_depth, num_channels, kernal_size):
        stddev = np.sqrt(2.0 / (num_channels * kernal_size[0] * kernal_size[1]))
        weights = cp.random.randn(kernal_depth, num_channels, kernal_size[0], kernal_size[1]) * stddev
        return weights

    def forward_pass(self, input_images):
        self.input_images = input_images
        
        self.output_images=conv2d(input_images,self.kernals,1,"valid","corr")
        self.output_images = self.leaky_relu(self.output_images + self.biases)

    def backward_pass(self, error_images):
        # Reshape error for leaky ReLU derivative
        error_images = error_images * self.leaky_relu_derivative(self.output_images)


                # Compute gradients using a fully vectorized cross-correlation operation
        batch_size = self.input_images.shape[0]
        in_channels = self.input_images.shape[1]
        out_channels = error_images.shape[1]

        # Reshape input for batch matmul compatible dimensions
        # For each spatial position in the output, we need a patch from the input
        # Extract im2col-style patches from input images
        input_patches = self.extract_patches(self.input_images,self.kernal_size)  # shape: (batch, out_h, out_w, in_channels, kernel_h, kernel_w)
        input_patches = input_patches.reshape(batch_size, -1, in_channels * self.kernal_size[0] * self.kernal_size[1])

        # Reshape error gradients to be compatible for multiplication
        errors_reshaped = error_images.reshape(batch_size, out_channels, -1)  # shape: (batch, out_channels, out_h * out_w)
        errors_reshaped = np.transpose(errors_reshaped, (0, 2, 1))  # shape: (batch, out_h * out_w, out_channels)

        # Perform batch matrix multiplication
        # Each batch element produces gradient contributions
        gradients_batch = np.matmul(input_patches.transpose(0, 2, 1), errors_reshaped)  # shape: (batch, in_ch*k_h*k_w, out_channels)

        # Sum over batch dimension and reshape to kernel dimensions
        self.kernel_gradients = np.sum(gradients_batch, axis=0)  # shape: (in_ch*k_h*k_w, out_channels)
        self.kernel_gradients = self.kernel_gradients.reshape(in_channels, self.kernal_size[0], self.kernal_size[1], out_channels)
        self.kernel_gradients = np.transpose(self.kernel_gradients, (3, 0, 1, 2))  # shape: (out_channels, in_channels, kernel_h, kernel_w)

        # Normalize by batch size


        # Calculate input derivatives using full convolution
        self.input_derivatives = conv2d(
            error_images,
            self.kernals,
            stride=1,
            padding="full",
            mode="conv"
        )

        # Calculate bias gradients - sum over spatial dimensions and average over batch
        self.bias_gradients = cp.mean(
            cp.sum(error_images, axis=(2, 3), keepdims=True),
            axis=0
        )

        return self.input_derivatives
    def apply_gradients(self,learn_rate=0.01):
        self.kernals-=self.kernal_gradients*learn_rate
        self.biases-=self.bias_gradients*learn_rate
    def leaky_relu(self, x):
        leak = 0.01
        return cp.maximum(x, leak * x)
    
    def leaky_relu_derivative(self, x):
        leak = 0.01
        return cp.where(x >= 0, 1.0, leak)

    def extract_patches(self,images, kernel_size, stride=1, padding=0):
        """
        Fully vectorized implementation to extract patches from input images.
        
        Args:
            images: Input tensor of shape (batch, channels, height, width)
            kernel_size: Tuple of (kernel_height, kernel_width)
            stride: Stride of the convolution (default: 1)
            padding: Padding size (default: 0)
            
        Returns:
            Patches of shape (batch, out_height, out_width, channels, kernel_height, kernel_width)
        """
        # Get input dimensions
        batch_size, n_channels, height, width = images.shape
        kernel_h, kernel_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        # Apply padding if specified
        if padding > 0:
            padded_images = cp.pad(
                images,
                pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode='constant',
                constant_values=0
            )
        else:
            padded_images = images
        
        # Calculate output dimensions
        padded_height, padded_width = padded_images.shape[2], padded_images.shape[3]
        out_height = (padded_height - kernel_h) // stride + 1
        out_width = (padded_width - kernel_w) // stride + 1
        
        # Create indices for batch and channel dimensions (these remain unchanged)
        batch_indices = cp.arange(batch_size)[:, cp.newaxis, cp.newaxis, cp.newaxis, cp.newaxis, cp.newaxis]
        channels_indices = cp.arange(n_channels)[cp.newaxis, cp.newaxis, cp.newaxis, :, cp.newaxis, cp.newaxis]
        
        # Create indices for height dimension
        h_indices = cp.arange(kernel_h)[cp.newaxis, cp.newaxis, cp.newaxis, cp.newaxis, :, cp.newaxis]
        h_pos = cp.arange(0, out_height * stride, stride)[:, cp.newaxis, cp.newaxis, cp.newaxis, cp.newaxis]
        h_indices = h_indices + h_pos
        
        # Create indices for width dimension
        w_indices = cp.arange(kernel_w)[cp.newaxis, cp.newaxis, cp.newaxis, cp.newaxis, cp.newaxis, :]
        w_pos = cp.arange(0, out_width * stride, stride)[cp.newaxis, :, cp.newaxis, cp.newaxis, cp.newaxis]
        w_indices = w_indices + w_pos
        
        # Gather all patches at once using advanced indexing
        patches = padded_images[
            batch_indices, 
            channels_indices,
            h_indices.reshape(1, out_height, 1, 1, kernel_h, 1).repeat(out_width, axis=2),
            w_indices.reshape(1, 1, out_width, 1, 1, kernel_w).repeat(out_height, axis=1)
        ]
        
        return patches.reshape(batch_size, out_height, out_width, n_channels, kernel_h, kernel_w)
        
class NeuralLayer:
    def __init__(self, number_of_neurons, previous_layer,batch_size=100):
        self.batch_size=batch_size
        self.number_of_neurons = number_of_neurons
        self.neurons = cp.zeros((batch_size,number_of_neurons))
        self.biases = cp.zeros(number_of_neurons)
        self.node_derivatives = cp.zeros((batch_size,number_of_neurons))
        
        if previous_layer is not None:
            self.previous_layer = previous_layer
            previous_layer.next_layer = self
            previous_layer.weights = self.he_initialization(previous_layer.number_of_neurons, number_of_neurons)
            previous_layer.gradients = cp.zeros((previous_layer.number_of_neurons, number_of_neurons))
    
    def he_initialization(self, n_in, n_out):
        stddev = np.sqrt(2.0 / n_in)
        weights = cp.random.randn(n_in, n_out) * stddev
        return weights
    
    def forward_pass(self):
        if hasattr(self, 'previous_layer') and self.previous_layer is not None:
            z = cp.dot(self.previous_layer.neurons, self.previous_layer.weights) + self.biases
            self.neurons = self.leaky_relu(z)
    
    def leaky_relu(self, x):
        leak = 0.01
        return cp.maximum(x, leak * x)
    
    def leaky_relu_derivative(self, x):
        leak = 0.01
        return cp.where(x >= 0, 1.0, leak)
    
    def input_values(self, values):
        self.neurons = to_gpu(values)
    
    def apply_gradients(self, learn_rate=0.01):
        if hasattr(self, 'weights'):
            self.weights -= learn_rate * self.gradients
            self.biases-=learn_rate*cp.sum(self.node_derivatives,axis=0)/self.batch_size
            self.gradients.fill(0.0)
    def backward_pass(self, next_layer=None):
        if next_layer is None:
            # This is the output layer
            return
            
        # Calculate node derivatives using the next layer
        delta = cp.dot(next_layer.node_derivatives, next_layer.previous_layer.weights.T)
        self.node_derivatives = delta * self.leaky_relu_derivative(self.neurons)
        
        # Calculate weight gradients
        if hasattr(self, 'previous_layer') and self.previous_layer is not None:
            self.previous_layer.gradients = cp.sum(cp.outer(
                self.previous_layer.neurons,
                self.node_derivatives
            ),axis=0)
def softmax(x):
    e_x = cp.exp(x - cp.max(x))
    return e_x / cp.sum(e_x)


def create_network(batch_size):
    """Initialize network with specified batch size"""
    global conv_layers, neural_layers
    conv_layers = [
        ConvLayer((batch_size,1,28,28), 32),
        PoolLayer((batch_size,32,26,26)),
        ConvLayer((batch_size,32,13,13), 64),
        PoolLayer((batch_size,64,11,11))
    ]
    
    neural_layers = []
    neural_numbers = [1600,128,10]
    for i in range(len(neural_numbers)):
        neural_layers.append(
            NeuralLayer(
                neural_numbers[i], 
                neural_layers[i-1] if i != 0 else None,
                batch_size=batch_size
            )
        )

def forward_pass(input_images, batch_size):
    cp.get_default_memory_pool().free_all_blocks()
    
    reshaped_input = input_images.reshape(batch_size, 1, 28, 28)
    conv_layers[0].forward_pass(reshaped_input)
    
    for i in range(1, len(conv_layers)):
        conv_layers[i].forward_pass(conv_layers[i-1].output_images)
        if i > 1:
            conv_layers[i-2].output_images = None
    
    neural_layers[0].input_values(conv_layers[-1].output_images.reshape(batch_size, -1))
    conv_layers[-1].output_images = None
    
    for i in range(1, len(neural_layers)):
        neural_layers[i].forward_pass()
    
    neural_layers[-1].neurons = softmax(neural_layers[-1].neurons)
    return neural_layers[-1].neurons

def train(batch_indices, learn_rate, batch_size):
    cp.get_default_memory_pool().free_all_blocks()
    
    batch_images = to_gpu(training_images[batch_indices])
    batch_labels = training_labels[batch_indices]
    
    output = forward_pass(batch_images, batch_size)
    expected_output = expected_output_array(batch_labels, batch_size)
    
    neural_layers[-1].node_derivatives = output - expected_output
    
    for i in range(len(neural_layers) - 2, 0, -1):
        neural_layers[i].backward_pass(neural_layers[i + 1])
        neural_layers[i+1].node_derivatives = None
    
    input_derivatives = neural_layers[0].node_derivatives.reshape(batch_size, -1)
    for i in range(len(conv_layers)-1, -1, -1):
        conv_layers[i].backward_pass(input_derivatives)
        input_derivatives = conv_layers[i].input_derivatives
        conv_layers[i].input_derivatives = None
    
    for layer in neural_layers:
        layer.apply_gradients(learn_rate)
    for layer in conv_layers:
        layer.apply_gradients(learn_rate)

def test(samples, batch_size):
    num_correct = 0
    num_batches = (samples + batch_size - 1) // batch_size
    
    for j in range(num_batches):
        start_idx = j * batch_size
        end_idx = min((j + 1) * batch_size, samples)
        current_batch_size = end_idx - start_idx
        
        test_batch = to_gpu(test_images[start_idx:end_idx])
        batch_labels = test_labels[start_idx:end_idx]
        
        outputs = forward_pass(test_batch, current_batch_size)
        predictions = cp.argmax(outputs, axis=1).get()
        num_correct += cp.sum(predictions == batch_labels[:current_batch_size])
                
    return num_correct / samples

def train_network(epochs=5, batch_size=5, learn_rate=0.001):
    # Initialize network with specified batch size
    create_network(batch_size)
    
    indices = np.arange(len(training_images))
    for i in range(epochs):
        np.random.shuffle(indices)
        start_time = time.time()
        
        for j in range(0, len(training_images), batch_size):
            batch_indices = indices[j:j + batch_size]
            if len(batch_indices) < batch_size:
                continue
                
            train(batch_indices, learn_rate, batch_size)
            
        learn_rate *= 0.9999
        epoch_time = time.time() - start_time
        accuracy = test(100, batch_size)
        print(f"Epoch {i+1} completed in {round(epoch_time, 2)} seconds, Accuracy: {accuracy}")

    print("Ending Accuracy: " + str(test(1000, batch_size)))

def expected_output_array(answers, batch_size):
    expected_output = cp.zeros((batch_size, 10))
    for i, answer in enumerate(answers):
        expected_output[i, answer] = 1
    return expected_output
train_network(5)
#with open("digits.json", "w") as f:
#    json.dump(conv_layers[1].output_images[0].tolist(), f, indent=2)
