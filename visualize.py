import visdom
import numpy as np

class LinePlotter(object):
    def __init__(self, env_name="main"):
        self.vis = visdom.Visdom()#server='http://localhost',port='8000',proxy='https://proxy.onera:80')
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]),
                                    Y=np.array([y, y]), env=self.env, opts=dict(
                                    legend=[split_name],
                                    title=var_name,
                                    xlabel="Iters",
                                    ylabel=var_name
                                    ))
        else:
            self.vis.updateTrace(X=np.array([x, x]), Y=np.array([y, y]), env=self.env,
                                win=self.plots[var_name], name=split_name)
