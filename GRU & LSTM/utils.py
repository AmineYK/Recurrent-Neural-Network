import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import string
import unicodedata
from torch.distributions import Multinomial
import sys 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Token de padding (BLANK)
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
TAILLE_VOCA = len(LETTRES)+1
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '<PAD>' ##NULL CHARACTER
id2lettre[EOS_IX] = '<EOS>'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


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

class GRU(RNN):
    #  TODO:  Implémenter un GRU

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
        self.linear_Z = nn.Linear(self.dim_in+self.dim_lat , self.dim_lat,bias=False)
        self.linear_R = nn.Linear(self.dim_in+self.dim_lat , self.dim_lat,bias=False)
        # a revoir
        self.linear_T = nn.Linear(self.dim_in+self.dim_lat , self.dim_lat,bias=False)
        self.sigmoide = nn.Sigmoid()
        self.tanh = nn.Tanh() 

        # decoder
        self.decoder = nn.Linear(self.dim_lat,self.dim_out,bias=True)


    def hzero(self, batch_size):

        hzero = torch.zeros(batch_size , self.dim_lat)
        return hzero

    def one_step(self, x_i, h):

        # concatenate the input and h(t-1)
        concat_input = torch.concat((h,x_i),dim=1)

        z_t = self.sigmoide(self.linear_Z(concat_input))
        r_t = self.sigmoide(self.linear_R(concat_input))

        temp_concat = torch.concat(( (r_t*h) , x_i ),dim=1)
        h_output = (1 - z_t) * h + z_t*self.tanh(self.linear_T(temp_concat))

        return h_output


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


    def decode(self,h, type='many-to-one'):
        h_s = []
            
        # h is a list of h_i 
        for h_i in h:
            h_s.append(self.decoder(h_i))

        return torch.stack(h_s)

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


def get_y(x):
    y = x[1:]
    zero_row = torch.zeros(1, x.size(1), dtype=x.dtype)  # Crée un tensor de zéros avec la même dimension que x
    return torch.cat((y,zero_row), dim=0)


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