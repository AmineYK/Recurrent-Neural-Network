
import torch
from torch.utils.data import DataLoader,Subset
from textloader import TextDataset,CONTENU,pad_collate_fn,TAILLE_VOCA,PAD_IX,EOS_IX
from generate import generate_beam,generate
from utils import *
from modules import Embedder_RNN , LSTM
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOADING DATA <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

TAILLE_DATASET = 1000
ds = TextDataset(CONTENU)
subset_indices = torch.randint(1,1000,(TAILLE_DATASET,))  # Choisissez les indices que vous souhaitez inclure
ds = Subset(ds, subset_indices)
loader = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=32)


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LSTM TEST <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

dim_in = TAILLE_VOCA
dim_lat = 60
dim_out = TAILLE_VOCA
lstm = LSTM(dim_in,dim_lat,dim_out)
embedder_lstm = Embedder_RNN(TAILLE_VOCA+1,TAILLE_VOCA,lstm)


LEARNING_RATE = 1e-1
EPOCHS = 20
optimizer = torch.optim.Adam(params=embedder_lstm.parameters() ,lr=LEARNING_RATE)


losses_tr = []
for epoch in tqdm(range(EPOCHS)):
    temp_loss = []
    for x in loader:

        optimizer.zero_grad()

        # dans y il y'aura la meme sequence que x mais decalé d'un caractere
        y = get_y(x)
        xhat = embedder_lstm.embed(x)
        h,liste_h,c,liste_c = embedder_lstm(x)
        output_decoded = embedder_lstm.decode(liste_h)
        loss = maskedCrossEntropy(output_decoded,y,padcar=PAD_IX)
        
        loss.backward()
        optimizer.step()

        #with torch.no_grad():
        temp_loss.append(loss.item())

    losses_tr.append(np.mean(temp_loss))


sentence_generated = generate(embedder_lstm,EOS_IX, start="Amine", maxlen=200)
print(sentence_generated)

plt.plot(range(EPOCHS) , losses_tr)
plt.title("Loss LSTM")
plt.show()

