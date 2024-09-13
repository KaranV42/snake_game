# Linear_QNet Overview

`Linear_QNet` is a simple feedforward neural network (fully connected) used to approximate Q-values in a Q-learning algorithm. This is commonly applied in reinforcement learning environments, where the model predicts the expected future rewards for each action in a given state.

## Architecture

- **Input layer:** The input size is determined by the number of features representing the state.
- **Hidden layer:** The model contains a single hidden layer with a specified number of neurons.
- **Output layer:** The output size is the number of possible actions, and each output represents the predicted Q-value for that action.

The model consists of two linear layers:
- `linear1:` Fully connected layer from the input to the hidden layer.
- `linear2:` Fully connected layer from the hidden to the output layer.

The activation function used between layers is ReLU (Rectified Linear Unit), which introduces non-linearity.

## Save Functionality

The save function allows you to save the model parameters to a `.pth` file, which can be reloaded later for inference or further training.

# QTrainer Overview

The `QTrainer` class is used to train the `Linear_QNet` model. It implements the Q-learning update rule.

## Parameters

- **Model:** The neural network model being trained.
- **Learning Rate (lr):** The step size for updating the model’s weights during training.
- **Gamma (γ):** The discount factor, which determines how much future rewards are considered when updating Q-values.

## Training Step

In each training step:

1. The current state, next state, action taken, reward received, and whether the episode is done (i.e., terminal state) are passed to the function.
2. Predicted Q-values for the current state are computed.
3. The target Q-value is updated using the Q-learning formula:

   \[
   Q(s,a) = r + \gamma \max_{a'} Q(s',a')
   \]

   where:
   - \( r \) is the reward,
   - \( \gamma \) is the discount factor,
   - \( \max_{a'} Q(s', a') \) is the maximum predicted Q-value for the next state.

4. The loss between the predicted Q-values and the target values is computed using Mean Squared Error (MSE).
5. Backpropagation and optimization are performed to update the model’s weights.

