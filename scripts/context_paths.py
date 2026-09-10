"""Shared finite context transitions; numerical hypotheses, not cognitive lifetimes."""

import math

import numpy as np


def transition(contexts,rate,elapsed_sec):
    if not math.isfinite(elapsed_sec) or elapsed_sec<0:
        raise ValueError('expected finite nonnegative elapsed time')
    if contexts==1:
        return np.ones((1,1))
    survival=math.exp(-rate*contexts/(contexts-1)*elapsed_sec)
    return np.full((contexts,contexts),(1-survival)/contexts)+survival*np.eye(contexts)


def branches(mode,contexts,rate,active,used,elapsed,initial):
    matrix=transition(contexts,rate,elapsed)
    parents,states,resets,fresh,returning,probabilities=[],[],[],[],[],[]
    for parent,current in enumerate(active):
        if initial or mode=='stationary':
            choices=[(0,1.,False,initial,False)]
        elif mode=='renewal':
            choices=[(0,math.exp(-rate*elapsed),False,False,False),
                     (0,-math.expm1(-rate*elapsed),True,True,False)]
        else:
            count=used[parent]
            choices=[(state,matrix[current,state],False,False,state!=current)
                     for state in range(count)]
            if count<contexts:
                choices.append((count,(contexts-count)*matrix[current,count],False,True,False))
        for state,probability,reset,is_fresh,is_returning in choices:
            if probability<=0:continue
            parents.append(parent);states.append(state);resets.append(reset)
            fresh.append(is_fresh);returning.append(is_returning)
            probabilities.append(math.log(probability))
    return dict(parent=np.array(parents),state=np.array(states),reset=np.array(resets),
                fresh=np.array(fresh),returning=np.array(returning),log_transition=np.array(probabilities))
