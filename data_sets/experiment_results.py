import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

datasets = ['banknote', 'glass', 'iris', 'mnist', 'page-blocks', 'phoneme', 'letters', 'seeds']
methods  = ['kmeans', 'qshift', 'hdbscan', 'qshiftpp']
rowidx   = {ds:i for i,ds in enumerate(datasets)}
colidx   = {mt:i for i,mt in enumerate(methods)}

#fig1, axes1 = plt.subplots(nrows=len(datasets), ncols=len(methods), sharey="row")
#fig2, axes2 = plt.subplots(nrows=len(datasets), ncols=len(methods))

fig1 = plt.figure()

for filename in glob.iglob('data_sets/**/*.pkl', recursive=True):
    print(filename)
    fname = os.path.splitext(os.path.basename(filename))[0]
    print(fname)
    fname_parts = fname.split('_')
    ri, ci = rowidx[fname_parts[0]], colidx[fname_parts[1]] 
    with open(filename, 'rb') as input_file:
        fig = pickle.load(input_file)
        fig1.axes.append(fig.axes)
        plt.close(fig)
    plt.figure(fig1.number)
    break
