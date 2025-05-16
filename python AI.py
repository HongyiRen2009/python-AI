import numpy as np
import cupy as cp
from mnist import MNIST
import time
import json
# Load MNIST dataset
mndata = MNIST("MNIST_ORG")
training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Convert to numpy arrays and normalize
training_images = np.array(training_images) / 255.0
test_images = np.array(test_images) / 255.0
training_labels=np.array(training_labels)
test_labels=np.array(test_labels)



# Helper functions to transfer data between CPU and GPU
def to_gpu(data):
    if isinstance(data, np.ndarray):
        return cp.asarray(data)
    return data

def to_cpu(data):
    if isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return data

class Layer:
    def __init__(self, number_of_neurons, previous_layer,decay_rate=0.9):
        self.number_of_neurons = number_of_neurons
        # For batch processing, neurons will now be a 2D array [batch_size, number_of_neurons]
        self.neurons = None
        self.biases = cp.zeros(number_of_neurons)
        self.node_derivatives = None
        self.decay_rate=decay_rate
        self.initialize_adam()
        if previous_layer is not None:
            self.previous_layer = previous_layer
            previous_layer.next_layer = self
            previous_layer.weights = self.he_initialization(previous_layer.number_of_neurons, number_of_neurons)
            previous_layer.gradients = cp.zeros((previous_layer.number_of_neurons, number_of_neurons))
            
    def he_initialization(self, n_in, n_out):
        # Draw weights from a normal distribution with std = sqrt(2 / n_in)
        stddev = np.sqrt(2.0 / n_in)
        # Transpose the weight matrix for batch multiplication
        weights = cp.random.randn(n_in, n_out) * stddev
        return weights
    
    def calc_value(self, batch_size):
        if hasattr(self, 'previous_layer') and self.previous_layer is not None:
            # [batch_size, n_in] @ [n_in, n_out] -> [batch_size, n_out]
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
    
    def initialize_adam(self):
        if hasattr(self, 'weights'):
            self.m = np.zeros_like(self.weights)  # First moment for weights
            self.v = np.zeros_like(self.weights)  # Second moment for weights
            self.m_b = np.zeros_like(self.biases)  # First moment for biases
            self.v_b = np.zeros_like(self.biases)  # Second moment for biases
            self.t = 0  # Timestep

    def apply_gradients(self, learn_rate=0.001, batch_size=1, 
                            beta1=0.9, beta2=0.999, epsilon=1e-8):
        if hasattr(self, 'weights'):
            # Initialize if not already done
            if not hasattr(self, 'm'):
                self.initialize_adam()
                
            # Increment timestep
            self.t += 1
            
            # Scale gradients by batch size
            grad = self.gradients / batch_size
            bias_grad = cp.sum(self.node_derivatives, axis=0) / batch_size  # Add bias gradients
            
            # Update biased first moment estimate
            self.m = beta1 * self.m + (1 - beta1) * grad
            self.m_b = beta1 * self.m_b + (1 - beta1) * bias_grad
            
            # Update biased second moment estimate
            self.v = beta2 * self.v + (1 - beta2) * (grad ** 2)
            self.v_b = beta2 * self.v_b + (1 - beta2) * (bias_grad ** 2)
            
            # Correct bias
            m_corrected = self.m / (1 - beta1 ** self.t)
            v_corrected = self.v / (1 - beta2 ** self.t)
            m_b_corrected = self.m_b / (1 - beta1 ** self.t)
            v_b_corrected = self.v_b / (1 - beta2 ** self.t)
            
            # Update parameters
            self.weights -= learn_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
            self.biases -= learn_rate * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)
            
            # Reset gradients
            self.gradients.fill(0.0)
def softmax(x):
    # Apply softmax to each example in the batch
    e_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
    return e_x / cp.sum(e_x, axis=1, keepdims=True)

# Initialize network layers
layer_numbers = [784, 128, 64,32, 10]
layers = []
for i in range(len(layer_numbers)):
    layers.append(Layer(layer_numbers[i], layers[i-1] if i != 0 else None))

def forward_pass(batch_data, batch_size):
    # Initialize input layer with batch data
    layers[0].input_values(batch_data)
    
    # Forward propagate through each layer
    for i in range(1, len(layers)):
        layers[i].calc_value(batch_size)
    
    # Apply softmax to the output layer
    layers[-1].neurons = softmax(layers[-1].neurons)
    return layers[-1].neurons

def expected_output_array(answers, batch_size):
    # Create one-hot encoded batch of expected outputs
    expected_output = cp.zeros((batch_size, 10))
    for i, answer in enumerate(answers):
        expected_output[i, answer] = 1
    return expected_output

def calculate_batch_gradients(batch_indices, batch_size):
    # Get batch data
    batch_images = to_gpu(training_images[batch_indices])
    batch_labels = training_labels[batch_indices]
    
    # Initialize node derivatives for all layers
    for layer in layers:
        layer.node_derivatives = cp.zeros((batch_size, layer.number_of_neurons))
    
    # Forward pass
    output = forward_pass(batch_images, batch_size)
    expected_output = expected_output_array(batch_labels, batch_size)
    
    # Output layer error
    layers[-1].node_derivatives = output - expected_output
    
    # Backpropagation through the network
    for i in range(len(layers) - 2, 0, -1):
        current_layer = layers[i]
        next_layer = layers[i+1]
        previous_layer = current_layer.previous_layer
        
        # Calculate node_derivatives for each example in the batch
        # [batch_size, n_out] @ [n_out, n_in] -> [batch_size, n_in]
        delta = cp.dot(next_layer.node_derivatives, next_layer.previous_layer.weights.T)
        current_layer.node_derivatives = delta * current_layer.leaky_relu_derivative(current_layer.neurons)
        
        # Calculate gradients by summing over the batch
        # [n_in, batch_size] @ [batch_size, n_out] -> [n_in, n_out]
        gradients = cp.dot(previous_layer.neurons.T, current_layer.node_derivatives)
        previous_layer.gradients += gradients

def apply_batch_gradients(learn_rate, batch_size):
    for i in range(len(layers) - 1):
        layers[i].apply_gradients(learn_rate, batch_size)

def test(samples=100):
    num_correct = 0
    # Process test data in mini-batches to avoid memory issues
    batch_size = 100
    num_batches = (samples + batch_size - 1) // batch_size
    
    for j in range(num_batches):
        start_idx = j * batch_size
        end_idx = min((j + 1) * batch_size, samples)
        current_batch_size = end_idx - start_idx
        
        test_batch = to_gpu(test_images[start_idx:end_idx])
        batch_labels = test_labels[start_idx:end_idx]
        
        # Forward pass for the batch
        outputs = forward_pass(test_batch, current_batch_size)
        predictions = cp.argmax(outputs, axis=1).get()
        
        # Count correct predictions
        for i in range(current_batch_size):
            if predictions[i] == batch_labels[i]:
                num_correct += 1
                
    return num_correct / samples
def augment_data(flat_images, image_shape, angle_degs, scales, offsets):
    N = len(flat_images)
    H, W = image_shape

    # Stack and reshape to (N, H, W)
    images = cp.stack([img.reshape(H, W) for img in flat_images])  # shape (N, H, W)

    cx, cy = W / 2, H / 2

    # Create coordinate grid
    y, x = cp.meshgrid(cp.arange(H), cp.arange(W), indexing='ij')
    coords = cp.stack([x - cx, y - cy], axis=-1)  # (H, W, 2)
    coords = cp.broadcast_to(coords, (N, H, W, 2))  # (N, H, W, 2)

    # Prepare transform matrices
    angles = cp.deg2rad(angle_degs)
    cos_a, sin_a = cp.cos(angles), cp.sin(angles)
    # Include horizontal and vertical scaling
    scale_x, scale_y = scales[:, 0], scales[:, 1]

    rot_mats = cp.stack([
        cp.stack([cos_a * scale_x, sin_a * scale_y], axis=1),
        cp.stack([-sin_a * scale_x, cos_a * scale_y], axis=1)
    ], axis=1)
    coords_flat = coords.reshape(N, -1, 2)
    src_coords = cp.einsum('nij,nkj->nki', rot_mats, coords_flat)  # (N, H*W, 2)

    # Recenter and apply offset
    src_coords[..., 0] += cx + offsets[:, 0][:, None]
    src_coords[..., 1] += cy + offsets[:, 1][:, None]

    # Nearest neighbor sampling
    src_x = cp.clip(cp.rint(src_coords[..., 0]).astype(cp.int32), 0, W - 1)
    src_y = cp.clip(cp.rint(src_coords[..., 1]).astype(cp.int32), 0, H - 1)

    batch_idx = cp.arange(N)[:, None]
    sampled = images[batch_idx, src_y, src_x]  # shape (N, H*W)
    return sampled
def train_batch(batch_indices, learn_rate):
    batch_size = len(batch_indices)
    calculate_batch_gradients(batch_indices, batch_size)
    apply_batch_gradients(learn_rate, batch_size)

def train_network(epochs=5, batch_size=100, learn_rate=0.001):
     print("Starting Accuracy: " + str(test()))
    
     indices = np.arange(len(training_images))
     angles = cp.random.uniform(-20, 20, size=batch_size)
     scales = cp.random.uniform(0.7, 1.3, size=(batch_size, 2))
     offsets = cp.random.uniform(-5, 5, size=(batch_size, 2))
     for i in range(0,len(training_images),batch_size):

          batch_indices = indices[i:i + batch_size]
          if len(batch_indices) < batch_size:
                    continue
          training_images[batch_indices]=augment_data(to_gpu(training_images[batch_indices]),(28,28),angles,scales,offsets).get()
     for i in range(0,len(test_images),batch_size):
          
          batch_indices = indices[i:i + batch_size]
          if len(batch_indices) < batch_size:
                    continue
          test_images[batch_indices]=augment_data(to_gpu(test_images[batch_indices]),(28,28),angles,scales,offsets).get()
     for i in range(epochs):
          np.random.shuffle(indices)
          start_time = time.time()
          
          for j in range(0, len(training_images), batch_size):
               batch_indices = indices[j:j + batch_size]
               if len(batch_indices) < batch_size:
                    continue  # Skip last batch if it's smaller than batch_size
                    
               train_batch(batch_indices, learn_rate)
               
          learn_rate *= 0.9999
          epoch_time = time.time() - start_time
          accuracy = test()
          print(f"Epoch {i+1} completed in {round(epoch_time, 2)} seconds, Accuracy: {accuracy}")

     print("Ending Accuracy: " + str(test(1000)))

# Run the training
if __name__ == "__main__":
    train_network(20)
    # Convert to JSON-serializable format
    serializable_model = {}
    for i in range(len(layers)):
        if(i==len(layers)-1):
            serializable_model[str(i)]=([layers[i].biases.tolist()])
        else:
            serializable_model[str(i)]=([layers[i].biases.tolist(),layers[i].weights.tolist()])
    # Save to JSON
    with open("python_model.json", "w") as f:
        json.dump(serializable_model, f, indent=2)