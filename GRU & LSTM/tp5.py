
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from textloader import *
from generate import *
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

#  TODO: 
def maskedCrossEntropy(output, target, padcar=0,reduce='mean'):
    """
    :param output: Tenseur length x batch x output_dim
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """

    # calculer la log-probabilité pour les étiquettes cibles
    log_probs = -torch.log_softmax(output, dim=2)
    
    # créer un masque binaire où les éléments égaux à padcar sont mis à zéro
    mask = (target != padcar).unsqueeze(2).float()
    
    # appliquer le masque aux log-probabilités
    masked_log_probs = log_probs * mask
    
    # calculer la somme des log-probabilités pour chaque élément
    sum_log_probs = torch.sum(masked_log_probs, dim=2)
    
    # calculer la somme des coûts pour chaque séquence
    sequence_costs = torch.sum(sum_log_probs, dim=0)
    
    # dans le cas ou on veut retoruner les loss pour chaque sequence
    if reduce == 'none':
        return sequence_costs
    
    # la loss moyenne du batch de sequences
    return torch.mean(sequence_costs)

class RNN(nn.Module):
    
    def __init__(self,dim_in,dim_lat,dim_out) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_lat = dim_lat
        self.dim_out = dim_out


        self.rnn_i = nn.Linear(self.dim_in,self.dim_lat,bias=False)
        self.rnn_h = nn.Linear(self.dim_lat,self.dim_lat,bias=True)
        self.rnn_tanh = nn.Tanh() 
        self.rnn_d = nn.Linear(self.dim_lat,self.dim_out,bias=True)
    
    def hzero(self,batch_size):
        # h0 : random initialisation with size (batch x latent_dim)
        return torch.zeros(batch_size , self.dim_lat)

    def one_step(self,x_i,h):

        return self.rnn_tanh(self.rnn_i(x_i) + self.rnn_h(h))

    def forward(self,x):

        # x --> full sequence
    
        length_seq = x.size(0)
        batch_size = x.size(1)

        h0 = self.hzero(batch_size)

        h_i = h0
        historical_h = []

        for i in range(length_seq):

            h_i = self.one_step(x[i] , h_i)
            historical_h.append(h_i)

        return h_i , historical_h 

    def decode(self,h,type='many-to-one'):

        output = self.rnn_d(h)
        # classification 
        if type == 'many-to-one':
            output_final = output

        if type == 'many-to-many':

            h_s = []
            
            # h is a list of h_i 
            for h_i in h:
                h_s.append(self.rnn_d(h_i))

            output_final = torch.stack(h_s)
            
        if type == 'many-to-many TG':
            
            h_s = []

            # h is a list of h_i 
            for h_i in h:
                h_s.append(self.rnn_d(h_i))

            output_final = torch.stack(h_s).softmax(dim=2)
            multinomial_dist = Multinomial(1, probs=output_final)
            x_i = multinomial_dist.sample()
            caracteres = torch.argmax(x_i,dim=2)
            output_final = torch.eye(TAILLE_VOCA)[caracteres]

        return output_final


class LSTM(RNN):
    def __init__(self, dim_in, dim_lat, dim_out) -> None:
        super().__init__(dim_in, dim_lat, dim_out)

        # params
        self.dim_in = dim_in
        self.dim_lat = dim_lat
        self.dim_out = dim_out
        
        # architecture
        '''the input dim is[self.dim_in+self.dim_lat] because 
        when we forward on it we will forward with the concatenation of the h_t-1 and x_t '''
        # self.dim_in+self.dim_lat --> self.dim_lat
        self.gate_forget = nn.Linear(self.dim_in+self.dim_lat , self.dim_lat,bias=True)
        self.gate_input = nn.Linear(self.dim_in+self.dim_lat , self.dim_lat,bias=True)
        self.gate_output = nn.Linear(self.dim_in+self.dim_lat , self.dim_lat,bias=True)
        self.gate_memory = nn.Linear(self.dim_in+self.dim_lat , self.dim_lat,bias=True)
        self.sigmoide = nn.Sigmoid()
        self.tanh = nn.Tanh() 

        # decoder
        self.decoder = nn.Linear(self.dim_lat,self.dim_out,bias=True)



    def hzero(self, batch_size):

        hzero = torch.zeros(batch_size , self.dim_lat)
        czero = torch.zeros(batch_size , self.dim_lat)
        return hzero , czero
    
    def one_step(self, x_i, h,c):

        # concatenate the input and h(t-1)
        concat_input = torch.concat((h,x_i),dim=1)

        f_t = self.sigmoide(self.gate_forget(concat_input))
        i_t = self.sigmoide(self.gate_input(concat_input))

        # term to term product
        c_output = f_t*c + i_t*self.tanh(self.gate_memory(concat_input))

        o_t = self.sigmoide(self.gate_output(concat_input))

        h_output = o_t * self.tanh(c_output)
        return h_output , c_output


    def forward(self, x):
        
        # x --> full sequence
    
        length_seq = x.size(0)
        batch_size = x.size(1)

        h0,c0 = self.hzero(batch_size)

        h_i = h0
        c_i = c0
        historical_h = []
        historical_c = []

        for i in range(length_seq):

            h_i,c_i = self.one_step(x[i] , h_i,c_i)
            historical_h.append(h_i)
            historical_c.append(c_i)

        return h_i , torch.stack(historical_h) , c_i , torch.stack(historical_c)

    def decode(self,h, type='many-to-one'):
        h_s = []
            
        # h is a list of h_i 
        for h_i in h:
            h_s.append(self.decoder(h_i))

        return torch.stack(h_s)

    

class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    pass


#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot


class Embedder_RNN(nn.Module):
    
    def __init__(self,dim_in,taille_voca,model_rnn):
        
        super(Embedder_RNN, self).__init__()
        
        self.taille_voca = taille_voca
        self.dim_in = dim_in
        
        #self.embedder = nn.Linear(self.dim_in ,self.taille_voca)
        self.embedder = torch.nn.Embedding(self.dim_in,self.taille_voca)
        self.rnn = model_rnn
        
        
    def forward(self, x):
        return self.rnn(self.embedder(x))

    def embed(self,x):
        return self.embedder(x)

    def decode(self,liste_h):
        return self.rnn.decode(liste_h)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOADING DATA <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

TAILLE_DATASET = 100
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

        with torch.no_grad():
                temp_loss.append(loss.item())

    losses_tr.append(np.mean(temp_loss))


sentence_generated = generate(embedder_lstm,EOS_IX, start="Amine", maxlen=200)
print(sentence_generated)

plt.plot(range(EPOCHS) , losses_tr)
plt.title("Loss LSTM")
plt.show()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< GRU TEST <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

