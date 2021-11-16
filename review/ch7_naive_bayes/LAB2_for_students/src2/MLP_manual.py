import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class BP:
    def __init__(self):
        self.input = torch.zeros((100, 5))  # sample size
        self.hidden_layer_1 = torch.zeros((100, 4))
        self.hidden_layer_2 = torch.zeros((100, 4))
        self.output_layer = torch.zeros((100, 3))
        self.w1 = (2 * torch.randn((5, 4)) - 1).requires_grad_(True)  # random matrix limited to (-1, 1)
        self.w2 = (2 * torch.randn((4, 4)) - 1).requires_grad_(True)
        self.w3 = (2 * torch.randn((4, 3)) - 1).requires_grad_(True)
        # store auto-grad result
        self.dw1_auto = None
        self.dw2_auto = None
        self.dw3_auto = None
        # store manual-grad result
        self.dw1_manual = None
        self.dw2_manual = None
        self.dw3_manual = None
        self.error = torch.zeros(3)
        self.learning_rate = 0.001

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def loss_func(self, y, pred):
        return - torch.sum(y * torch.log(pred))

    def fp(self, data, label):  # label:100 X 3,data: 100 X 5
        self.input = data
        self.hidden_layer_1 = torch.sigmoid(torch.mm(self.input, self.w1))
        self.hidden_layer_2 = torch.sigmoid(torch.mm(self.hidden_layer_1, self.w2))
        self.output_layer = torch.softmax(torch.mm(self.hidden_layer_2, self.w3), dim=1)
        # error
        self.error = self.output_layer - label
        loss = self.loss_func(label, self.output_layer)
        loss.backward()
        self.dw1_auto = self.w1.grad
        self.dw2_auto = self.w2.grad
        self.dw3_auto = self.w3.grad
        # print("w1.grad:", self.w1.grad)
        # print("w2.grad:", self.w2.grad)
        # print("w3.grad:", self.w3.grad)
        return loss

    def bp(self):
        output_diff = self.error
        hidden_diff_2 = torch.mm(output_diff, self.w3.T) * self.sigmoid_derivative(self.hidden_layer_2)
        hidden_diff_1 = torch.mm(hidden_diff_2, self.w2.T) * self.sigmoid_derivative(self.hidden_layer_1)
        grad_w3 = torch.mm(self.hidden_layer_2.T, output_diff)
        grad_w2 = torch.mm(self.hidden_layer_1.T, hidden_diff_2)
        grad_w1 = torch.mm(self.input.T, hidden_diff_1)
        self.dw1_manual = grad_w1
        self.dw2_manual = grad_w2
        self.dw3_manual = grad_w3
        # print("grad_w1", grad_w1)
        # print("grad_w2", grad_w2)
        # print("grad_w3", grad_w3)
        # update
        # print("1.", self.w1.is_leaf)
        self.w3 = self.w3 - self.learning_rate * torch.mm(self.hidden_layer_2.T, output_diff)
        self.w2 = self.w2 - self.learning_rate * torch.mm(self.hidden_layer_1.T, hidden_diff_2)
        self.w1 = self.w1 - self.learning_rate * torch.mm(self.input.T, hidden_diff_1)
        # print("2.", self.w1.is_leaf)
        # reset grad zero by assignment instead of grad.data.zero_
        self.w1 = torch.tensor(self.w1, requires_grad=True)
        self.w2 = torch.tensor(self.w2, requires_grad=True)
        self.w3 = torch.tensor(self.w3, requires_grad=True)


# from torchvision load data
def generate_data():
    # generate random features
    X = torch.randn((100, 5))
    # generate label
    y = []
    for i in range(33):
        y.append(0)
    for i in range(33, 66):
        y.append(1)
    for i in range(66, 100):
        y.append(2)
    y = torch.tensor(y)
    return X, y


def convert_to_one_hot(y, C):
    return torch.eye(C)[y.reshape(-1)]


def bp_network():
    nn = BP()
    X, y = generate_data()
    y = convert_to_one_hot(y, 3)
    epochs = 1000
    loss_list = []
    for i, epoch in enumerate(range(epochs)):
        loss = nn.fp(X, y)/100
        loss_list.append(loss)
        nn.bp()
        # examine deri every 100 times
        if i % 100 == 0:
            print("time:" + str(i))

            print("dw1_manual:")
            print(nn.dw1_manual)
            print("dw1_auto:")
            print(nn.dw1_auto)

            print("dw2_manual:")
            print(nn.dw2_manual)
            print("dw2_auto:")
            print(nn.dw2_auto)

            print("dw3_manual:")
            print(nn.dw3_manual)
            print("dw3_auto:")
            print(nn.dw3_auto)
            print('\n')

    loss_x = [i for i in range(epochs)]
    print(f'final loss:{loss_list[-1]}')
    plt.plot(loss_x, loss_list)
    plt.show()
    return nn


if __name__ == '__main__':
    bp_network()
