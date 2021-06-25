from bokeh.models.markers import X
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_notebook




def disp_2_func(func1="Linear", func2="Linear"):
    p = figure(width=700, height=500, x_axis_label="Input", y_axis_label="Output", x_range=(-5, 5), y_range=(-2, 2))
    x = np.arange(-10, 10, 0.01)

    ref = {    
        "Tanh": np.tanh(x),
        "Sigmoid": 1/(1+np.exp(-x)),
        "ReLU": np.maximum(0, x),
        "Leaky ReLU": np.maximum(0.1*x, x),
        "Linear": x,
        "ELU": np.maximum(0, x) + np.minimum(np.exp(x)-1, 0)
    }
    try:
        y1 = ref[func1]
    except KeyError:
        y1 = ref["Linear"]

    try:
        y2 = ref[func2]
    except KeyError:
        y2 = ref["Linear"]




    p.line([0, 0], [100, -100], color="black")
    p.line([100, -100], [0, 0], color="black")

    p.line(x, y1, color="blue", legend_label=func1)
    p.line(x, y2, color="red", legend_label=func2)

    output_notebook()
    p.legend.title = "Activation Functions"
    show(p)

def sigmoid(z):
    return 1/(1 + np.exp(-z))