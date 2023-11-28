import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, input_channels):
        super(Linear_QNet, self).__init__()

        # Define the CNN layers for snake positions
        self.snake_cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Define the CNN layers for food positions
        self.food_cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Fully connected layer for other features (e.g., game state)
        self.linear = nn.Linear(input_size, hidden_size)

        # Fully connected layers for Q-values
        self.fc1 = nn.Linear(32 * 11 * 11, hidden_size)  # Assuming 11x11 grid and 32 channels
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, snake_positions, food_positions, other_features):
        # Forward pass for snake positions
        snake_x = self.snake_cnn(snake_positions)
        snake_x = snake_x.view(snake_x.size(0), -1)

        # Forward pass for food positions
        food_x = self.food_cnn(food_positions)
        food_x = food_x.view(food_x.size(0), -1)

        # Concatenate the flattened outputs
        x = torch.cat((snake_x, food_x), dim=1)

        # Fully connected layer for other features
        other_x = F.relu(self.linear(other_features))

        # Concatenate the outputs of CNN and fully connected layers
        x = torch.cat((x, other_x), dim=1)

        # Fully connected layers for Q-values
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, snake_positions, food_positions, next_snake_positions, next_food_positions):
        # Convert input data to PyTorch tensors
        snake_positions = torch.tensor(snake_positions, dtype=torch.float)
        food_positions = torch.tensor(food_positions, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_snake_positions = torch.tensor(next_snake_positions, dtype=torch.float)
        next_food_positions = torch.tensor(next_food_positions, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)  # Add next_state

        # Handle the case where state is a 1D array
        if len(snake_positions.shape) == 1:
            snake_positions = torch.unsqueeze(snake_positions, 0)
            food_positions = torch.unsqueeze(food_positions, 0)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_snake_positions = torch.unsqueeze(next_snake_positions, 0)
            next_food_positions = torch.unsqueeze(next_food_positions, 0)
            next_state = torch.unsqueeze(next_state, 0)  # Add next_state
            done = (done, )

        # Concatenate snake, food, and other positions
        con_state = (snake_positions, food_positions, state)

        # 1: predicted Q values with current state
        pred = self.model(*con_state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                con_next_state = (next_snake_positions[idx], next_food_positions[idx], next_state[idx])  # Add next_state
                Q_new = reward[idx] + self.gamma * torch.max(self.model(*con_next_state))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # loss value
        loss_value = loss.item()
        print(f"Loss: {loss_value}")

        self.optimizer.step()



    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        # loss value
        loss_value = loss.item()
        print(f"Loss: {loss_value}")

        self.optimizer.step()



