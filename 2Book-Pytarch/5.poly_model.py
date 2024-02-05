import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim import optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x**i for i in range(1, 4)], 1)
    # 矩阵的拼接


W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

def f(x):
    return x.mm(W_target) + b_target[0]

def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x.to(device), y.to(device)

class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out

if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
while True:
    batch_x, batch_y = get_batch()

    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    epoch += 1
    if print_loss < 1e-3:
        break

print(W_target)
print(b_target)
