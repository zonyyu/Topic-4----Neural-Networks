import torch
from torch.nn import Module, Linear, Softmax, ReLU, BatchNorm1d, CrossEntropyLoss, Dropout
from torch.optim import SGD
from torchmetrics import Accuracy

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from PIL import Image


model = torch.jit.load("mnist_predictor.pt")
model.eval()



def predict_nums(lst):

    proc_img = []
    for i in range(len(lst)):
        nparr = np.sum(lst[i, :, :, :3], axis=-1)/3       
        img = Image.fromarray(nparr).resize((28, 28))
        nparr = np.maximum(np.minimum(np.array(img), 255), 0)
        proc_img.append(nparr.flatten())
    proc_img = np.array(proc_img)
    #sn.heatmap(proc_img[0].reshape(28, 28), annot=True)
    #plt.show()

    inputs = torch.from_numpy(proc_img).float()
    pred = model(inputs).detach().numpy()
    pred = np.argmax(pred, axis=1)
    print(pred)



