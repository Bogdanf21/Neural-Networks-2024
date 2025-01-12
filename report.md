# Deep Q-Network (DQN) Implementation for Flappy Bird

## Overview
This repository contains the implementation of a Deep Q-Network (DQN) to train an agent for playing Flappy Bird. The model leverages convolutional neural networks to process game frames and reinforcement learning principles to optimize gameplay decisions.

## Environment
The environment was taken from https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/game/wrapped_flappy_bird.py since the environment provided in the Tensor-Reloaded repository was unfortunately too slow to train on (regradless if the game was displayed or not)
---

## Hyperparameters
The following hyperparameters govern the training process and model behavior:

| Parameter                 | Value           | Description |
|---------------------------|-----------------|-------------|
| `MODEL_NAME`              | `Flapflaptry26` | Model identifier for saved checkpoints |
| `LEARNING_RATE`           | `1e-5`          | Learning rate for the optimizer |
| `FRAME_SKIP`              | `1`             | Frames skipped during action repetition |
| `FRAME_SKIP_JUMP`         | `0`             | Additional frames skipped for jump actions |
| `SHOW_GAME`               | `True`          | Whether to render the game during training/test |
| `NUMBER_OF_ACTIONS`       | `2`             | Number of available actions (flap or no action) |
| `GAMMA`                   | `0.99`          | Discount factor for future rewards |
| `INITIAL_EPSILON`         | `0.2`           | Starting exploration rate |
| `FINAL_EPSILON`           | `0.00001`       | Final exploration rate |
| `NUMBER_OF_ITERATIONS`    | `2000000`       | Total iterations for training |
| `REPLAY_MEMORY_SIZE`      | `50000`         | Maximum size of the replay memory |
| `MINIBATCH_SIZE`          | `32`            | Batch size for optimization |
| `TARGET_UPDATE_FREQUENCY` | `1000`          | Frequency for updating the target network |

---

## Neural Network Architecture
The agent's decision-making is powered by a convolutional neural network (CNN) defined as follows:

### Architecture

```python
class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, 4)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 4, 2) # Second convolutional layer
        self.conv3 = nn.Conv2d(64, 64, 3, 1) # Third convolutional layer
        self.fc4 = nn.Linear(3136, 512)      # Fully connected layer
        self.fc5 = nn.Linear(512, 2)         # Output layer for actions

    def forward(self, x):
        output = self.conv1(x)
        output = torch.nn.functional.relu(output)
        output = self.conv2(output)
        output = torch.nn.functional.relu(output)
        output = self.conv3(output)
        output = torch.nn.functional.relu(output)
        output = output.view(output.size()[0], -1)
        output = self.fc4(output)
        output = torch.nn.functional.relu(output)
        output = self.fc5(output)

        return output
```

### Layer Details
1. **Convolutional Layers**:
   - Extract spatial and temporal features from the input frames.
   - `conv1`: Input channels = 4 (stacked frames), filters = 32, kernel size = 8, stride = 4.
   - `conv2`: Input channels = 32, filters = 64, kernel size = 4, stride = 2.
   - `conv3`: Input channels = 64, filters = 64, kernel size = 3, stride = 1.

2. **Fully Connected Layers**:
   - `fc4`: Compresses features into a 512-dimensional vector.
   - `fc5`: Maps the features to the two possible actions (flap or no action).

### Activation Functions
ReLU (Rectified Linear Unit) is applied after each convolutional and fully connected layer to introduce non-linearity.

### Initialization
Weights and biases are initialized using a uniform distribution:

```python
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)
```

---

## Image Preprocessing
The game frames are preprocessed to a standardized input format:

1. Cropped to remove unnecessary regions.
2. Resized to an 84x84 grayscale image.
3. Normalized to values between 0 and 1.
4. Converted to a tensor compatible with PyTorch.

```python
def image_processing(image):
    image = image[:, 40:300]  # Crop irrelevant parts
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    image_tensor = image_data.transpose(2, 0, 1).astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor
```

---

## Training Procedure
The training algorithm follows the DQN framework:

1. **Replay Memory**:
   - Stores past experiences (`state`, `action`, `reward`, `next_state`, `done`).
   - Size limited to `REPLAY_MEMORY_SIZE`.

2. **Optimization**:
   - Samples a minibatch of size `MINIBATCH_SIZE` from replay memory.
   - Computes target Q-values using the target network.
   - Updates the main network using Mean Squared Error (MSE) loss.

3. **Exploration vs. Exploitation**:
   - Epsilon-greedy strategy adjusts exploration rate (`epsilon`) over time, from `INITIAL_EPSILON` to `FINAL_EPSILON`.

4. **Target Network**:
   - Periodically updated to improve training stability.

5. **Checkpointing**:
   - Saves the model every 25,000 iterations.

---

## Testing
During testing, the trained model is evaluated without exploration. The agent selects actions based solely on the learned policy:

```python
def test(model):
    game_state = GameState(Hyperparameters.SHOW_GAME)

    action = torch.zeros([Hyperparameters.NUMBER_OF_ACTIONS], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = image_processing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        output = model(state)[0]
        action = torch.zeros([Hyperparameters.NUMBER_OF_ACTIONS], dtype=torch.float32)
        action_index = torch.argmax(output).item()
        action[action_index] = 1

        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = image_processing(image_data_1)
        new_state = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        state = new_state
```

---

## Experimentations
The following experiments were conducted to improve the model's performance:

1. **Additional Dense Layers**:
   - Two dense layers of 512 neurons were added between the convolutional layers and the output layer.
   - Result: The model underfitted, unable to learn meaningful patterns from the data.

2. **MaxPooling Layers**:
   - MaxPooling layers were introduced between the convolutional layers to enhance feature extraction.
   - Result: The model overfitted on specific paths, repeatedly choosing the same actions regardless of the gap position.

3. **Increased Batch Size**:
   - Larger batch sizes were tested to improve training stability.
   - Result: The model's convergence was significantly slower.

4. **Frame Stacks**:
   - Experiments were conducted with frame stacks of 2, 3, and 4.
   - Result: A stack size of 4, while slower to converge, provided more stable and reliable performance compared to smaller stack sizes.

---

## Execution
- **Training**: Run the `train()` function to start the training process.
- **Testing**: Load a trained model checkpoint and evaluate it using the `test()` function.
