from PS_NN import *
import numpy
import os
import random

t_0 = time.clock()
NN = PS_NeuralNet("","","")
i = [0,0,0,0,0,0,0,0]

txt_files = []
folder = "C:/Users/Johanan/Documents/Python/PS_NN/training_data"
for file in os.listdir(folder):
    txt_files.append(os.path.join(folder,file))



for a in range(2):
    random.shuffle(txt_files)
    print("Done shuffle: " + str(a))
    for filepath in txt_files:
        f = open(filepath,'r')
        content = f.read()
        uni_text = content[0:8]
        PS_text = content[8:]
        PS_text = PS_text.replace("\n","")
        if len(PS_text) > 4000:
            PS_text = PS_text[8:4008]
        while len(PS_text) < 4000:
            PS_text += " "
        f.close()
        PS = np.matrix([(ord(c)-65)/10 for c in PS_text])
        uni = np.matrix(list([float(c)] for c in uni_text))
        
        #for n in range(8):
            #if uni[n] == 1:
             #   i[n] += 1
       # print(i)
        NN.train(PS,uni)

t_1 = time.clock() - t_0 
print("Training time: " + str(t_1))
print("Writing weights to file...")

NN.weights1.dump("weights1.dat")
t_2 = time.clock() - t_1
print("Writing time of weights1: " + str(t_2))

NN.weights2.dump("weights2.dat")
t_3 = time.clock() - t_2
print("Writing time of weights2: " + str(t_3))

NN.weights3.dump("weights3.dat")
t_4 = time.clock() - t_3
print("Writing time of weights3: " + str(t_4))
print("Total run time: " + str(t_4 + t_3 + t_2 + t_1))

