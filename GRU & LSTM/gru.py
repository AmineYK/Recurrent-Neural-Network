
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from textloader import *
from generate import *
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOADING DATA <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

TAILLE_DATASET = 1000
ds = TextDataset(CONTENU)
subset_indices = torch.randint(1,1000,(TAILLE_DATASET,))  # Choisissez les indices que vous souhaitez inclure
ds = Subset(ds, subset_indices)
loader = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=32)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< GRU TEST <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

dim_in = TAILLE_VOCA
dim_lat = 60
dim_out = TAILLE_VOCA
gru = GRU(dim_in,dim_lat,dim_out)
embedder_gru = Embedder_RNN(TAILLE_VOCA+1,TAILLE_VOCA,gru)

'''
LEARNING_RATE = 1e-1
EPOCHS = 20
optimizer = torch.optim.Adam(params=embedder_gru.parameters() ,lr=LEARNING_RATE)


losses_tr = []
for epoch in tqdm(range(EPOCHS)):
    temp_loss = []
    for x in loader:

        optimizer.zero_grad()

        # dans y il y'aura la meme sequence que x mais decalé d'un caractere
        y = get_y(x)
        xhat = embedder_gru.embed(x)
        h,liste_h = embedder_gru(x)
        output_decoded = embedder_gru.decode(liste_h)
        loss = maskedCrossEntropy(output_decoded,y,padcar=PAD_IX)
        
        loss.backward()
        optimizer.step()

        with torch.no_grad():
                temp_loss.append(loss.item())

    losses_tr.append(np.mean(temp_loss))'''


'''sentence_generated = generate(embedder_gru,EOS_IX, start="Amine", maxlen=200)
print(sentence_generated)'''

'''plt.plot(range(EPOCHS) , losses_tr)
plt.title("Loss LSTM")
plt.show()
'''
a = generate_beam(embedder_gru,EOS_IX,6, start="That is some group of people", maxlen=200)

b = generate_beam_nucleus(embedder_gru,EOS_IX,6,alpha=0.75, start="That is some group of people", maxlen=200)
