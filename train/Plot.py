import argparse
import matplotlib.pyplot as plt
import os 
import json 




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


## resume directory
parser.add_argument('--resume-dir-list', type=str, nargs='+', help='resume directory list')
parser.add_argument('--key-list', type=str, nargs='+', help='which key to plot, valmAP, trainAcc, trainPosAcc, trainNegAcc, valPosAcc, valNegAcc, bestPerf, trainLossCorr')
parser.add_argument('--label', type=str, nargs='+', help='label for each directory')
parser.add_argument('--ylabel', type=str, help='ylabel')

## out-name
parser.add_argument('--out-name', type=str, help='output plot name')
args = parser.parse_args()
print (args)

history = []
nb_line = len(args.resume_dir_list)
for i in range(nb_line) : 
    with open(os.path.join(args.resume_dir_list[i], 'history.json'), 'r') as f :
        history.append(json.load(f)[args.key_list[i]])

color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
max_epoch = 0
for i in range(nb_line) :
    
    plt.plot(history[i], '--', color= color_list[i], lw = 2, label=args.label[i])
    max_epoch = max(max_epoch, len(history[i]))

plt.xlabel('# Epoch')
plt.ylabel(args.ylabel)

plt.legend(loc='best')
plt.grid('on')
plt.savefig(args.out_name, bbox_inches = 'tight', pad_inches = 0)
