from collections import OrderedDict
from functools import reduce

from numpy import vstack
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.nn import BCELoss
from torch.nn import Parameter
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Sequential
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.optim import SGD
from torch.utils.data import DataLoader

from csv_dataset import CSVDataset


class LonosphereModel(Module):
    def __init__(self, n_inputs):
        super().__init__()
        # self._ex1 = Linear(n_inputs, 1)
        self._layer_names = ['hidden_1', 'act_1', 'hidden_2', 'act_2', 'hidden_3', 'act_3']
        self._layers = Sequential(OrderedDict([
            ('hidden_1', Linear(n_inputs, 10)),
            ('act_1', ReLU()),
            ('hidden_2', Linear(10, 8)),
            ('act_2', ReLU()),
            ('hidden_3', Linear(8, 1)),
            ('act_3', Sigmoid()),]))
        kaiming_uniform_(self._layers.hidden_1.weight, nonlinearity='relu')
        kaiming_uniform_(self._layers.hidden_2.weight, nonlinearity='relu')
        xavier_uniform_(self._layers.hidden_3.weight)
        
    def forward(self, x):
        return self._layers(x)
        # return Sigmoid()(self._ex1(x))
        # return reduce(lambda data, l_name: self._layers[l_name](data), x)


def prepare_data(p):
    dataset = CSVDataset(p)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


def train_model(train_dl, model):
    print(f'train {model}')
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(100):
        for i, (x_batch, y_batch) in enumerate(train_dl):
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()


def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (x_batch, y_batch) in enumerate(test_dl):
        y_hat = model(x_batch).detach().numpy()
        y_hat = y_hat.round()
        y = y_batch.numpy().reshape((-1, 1))
        predictions.append(y_hat)
        actuals.append(y)
    predictions = vstack(predictions)
    actuals = vstack(actuals)
    return accuracy_score(actuals, predictions)


def predict(row, model):
    row = Tensor(row)
    return model(row).detach().numpy()


def run():
    path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
    train_dl, test_dl = prepare_data(path)
    print(len(train_dl.dataset), len(test_dl.dataset))
    
    model = LonosphereModel(34)
    # for i, p in enumerate(model.parameters()): print(i, p)
    train_model(train_dl, model)
    
    acc = evaluate_model(test_dl, model)
    print('Accuracy: %.3f' % acc)

    row = [1, 0, 0.99539, -0.05889, 0.85243, 0.02306, 0.83398, -0.37708, 1, 0.03760, 0.85243, -0.17755, 0.59755,
           -0.44945, 0.60536, -0.38223, 0.84356, -0.38542, 0.58212, -0.32192, 0.56971, -0.29674, 0.36946, -0.47357,
           0.56811, -0.51171, 0.41078, -0.46168, 0.21266, -0.34090, 0.42267, -0.54487, 0.18641, -0.45300]
    y_hat = predict(row, model)[0]
    print(f'Predicted: {y_hat:.3f} (class={y_hat.round()})')


if __name__ == '__main__':
    run()
