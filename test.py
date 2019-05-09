import threading
from time import sleep
import numpy as np

if __name__ == "__main__":
    s = np.array([333])
    print(type(s[0]))
    print(type(s[0].item()))
    # print(s)
    # print(type(s))
