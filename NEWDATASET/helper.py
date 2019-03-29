from time import time
from random import random


def cp(t, string="", end = "-"*50):
    string += ": "
    now = time()
    print(end,string,round(now-t[-1],5),"seconds",end)
    t.append(now)

def end(t):
    now = time()
    print("end:",round(now-t[0],5),"seconds") 

def getClass(p, floor=0.4, ceiling=0.6, record=None):
    if record:
        record[1] += 1
        if p > floor and p < ceiling:
            record[0] += 1

    return 0 if p<floor else 1 if p>ceiling else 1 if random()<p else 0

