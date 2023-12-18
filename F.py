import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


def preprocess_data(df):

    df['Price'] = df['Price'].str.replace('$', '').astype(float)
    df['Shipping'] = df['Shipping'].str.replace('Free', '0').str.replace('$', '').astype(float)

   s
    features = df[['Price', 'Shipping']].values
    targets = df['Target'].values.reshape(-1, 1)

   
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    
    dataset = TensorDataset(features_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader


def train_neural_network(model, dataloader, epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


def visualize_predictions(model, dataloader):
    model.eval()
    actual_prices = []
    predicted_prices = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            actual_prices.extend(targets.numpy())
            predicted_prices.extend(outputs.numpy())

    sns.scatterplot(x=actual_prices, y=predicted_prices)
    plt.title('Actual Prices vs Predicted Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.show()

if __name__ == '__main__':

    df = pd.read_csv('ebay.csv')

r
    dataloader = preprocess_data(df)

   rk
    model = SimpleNN(input_size=2)  # Assuming 2 input features
    train_neural_network(model, dataloader)

 
    visualize_predictions(model, dataloader)
