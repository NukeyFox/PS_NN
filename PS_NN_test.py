from PS_NN import *

folder = "C:/Users/Johanan/Documents/Python/PS_NN/training_data"
filepath = os.path.join(folder,"PS_103.txt")
f = open(filepath,'r')
PS_text = f.read()
PS_text = PS_text.replace("\n","")
if len(PS_text) > 4000:
    PS_text = PS_text[8:4008]
while len(PS_text) < 4000:
    PS_text += " "
f.close()
PS = np.matrix([ord(c) for c in PS_text])

NN = PS_NeuralNet("weights1.dat","weights2.dat","weights3.dat")
print(NN.predict(PS))

#[Oxbridge, Imperial, UCL, Edinburgh, Southampton, Warrick, Durham, Nonsense]