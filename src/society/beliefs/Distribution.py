import numpy as np
import random

class Distribution:
    def __init__(self, type="unique", value = None, range=None):
        self.type = type
        self.value = value
        self.range = range


    def generate(self, size):
        match self.type:
            case "unique":
                return self.unique(size)
            case "uniform":
                return self.uniform(size)
            case "linespace":
                return self.linespace(size)
            
    def unique(self, size):
        return [self.value for i in range(size)]
    
    def uniform(self, size):
        return np.random.uniform(self.range[0], self.range[1], size)
    
    def linespace(self, size):
        linespace = np.linspace(self.range[0], self.range[1], size)
        np.random.shuffle(linespace)
        return linespace