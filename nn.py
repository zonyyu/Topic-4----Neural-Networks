import torch
from torch.nn import Module, Linear, Softmax, ReLU, BatchNorm1d, CrossEntropyLoss, Dropout
from torch.optim import SGD
from torchmetrics import Accuracy

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from PIL import Image


class NN(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.bn = BatchNorm1d(input_dim)
        self.z1 = Linear(input_dim, 900)
        self.a1 = ReLU()
        
        self.z2 = Linear(900, 900)
        self.a2 = ReLU()
        
        self.z3 = Linear(900, output_dim)
        
        self.sm = Softmax(dim=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.z1(x)
        x = self.a1(x)
        
        x = self.z2(x)
        x = self.a2(x)
        
        x = self.z3(x)
        x = self.sm(x)
        
        return x
    
    def fit(self, X_train, Y_train, loss_fn, opt, X_cv=None, Y_cv=None, epochs=1):
        for i in range(epochs):
            opt.zero_grad()
            self.train()
            Y_pred = self(X_train)
            
            cost = loss_fn(Y_pred, Y_train)
            
            acc = Accuracy().cuda()
            msg = f"Iter: {i+1}, loss: {cost.item(): .4f}, accuracy: {acc(torch.argmax(Y_pred, dim=1).int(), torch.argmax(Y_train, dim=1).int()): .4f}"
            
            if torch.is_tensor(X_cv) and torch.is_tensor(Y_cv):
                self.eval()
                Y_pred_cv = self(X_cv)
                val_cost = loss_fn(Y_pred_cv, Y_cv)
                msg += f",  val_loss: {val_cost.item(): .4f}, val_accuracy: {acc(torch.argmax(Y_pred_cv, dim=1).int(), torch.argmax(Y_cv, dim=1).int()): .4f}"
                self.train()
            cost.backward()
            print(msg)
            opt.step()
            






def CCE(Y_pred, Y_true):
    Y_pred = 0.9999 * Y_pred + (1-0.9999)/2
    
    ylogy = -Y_true * torch.log(Y_pred)
    class_sum = torch.sum(ylogy, axis=1)
    set_sum = torch.sum(class_sum)/Y_pred.shape[0]
    return set_sum
    


model = NN(784, 10)
model.load_state_dict(torch.load("mnist_predictor.pt"))
model.eval()



def predict_nums(lst):

    proc_img = []
    for i in range(len(lst)):
        nparr = np.sum(lst[i, :, :, :3], axis=-1)/3       
        img = Image.fromarray(nparr).resize((28, 28))
        nparr = np.maximum(np.minimum(np.array(img), 255), 0)
        proc_img.append(nparr.flatten())
    proc_img = np.array(proc_img)
    sn.heatmap(proc_img[0].reshape(28, 28), annot=True)
    plt.show()

    inputs = torch.from_numpy(proc_img).float()
    pred = model(inputs).detach().numpy()
    pred = np.argmax(pred, axis=1)
    print(pred)



