import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        '''self.rnn_i = nn.Sequential(
            nn.Linear(self.dim_in,int(self.dim_lat // 4 ) ),
            nn.Tanh(),
            nn.Linear(int(self.dim_lat // 4 ) , int(self.dim_lat // 2 ) ),
            nn.Tanh(),
            nn.Linear(int(self.dim_lat // 2 ) , self.dim_lat )
        ) 
        self.rnn_h = nn.Sequential(
            nn.Linear(self.dim_lat,int(self.dim_lat // 2 ) ),
            nn.Tanh(),
            nn.Linear(int(self.dim_lat // 2 ) , self.dim_lat )
        )
        self.rnn_tanh = nn.Tanh() 
        self.rnn_d = nn.Linear(self.dim_lat,self.dim_out,bias=True)'''
    
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

        output = self.rnn_tanh(self.rnn_d(h))
        # classification 
        if type == 'many-to-one':
            output_final = output

        if type == 'many-to-many':

            h_s = []
            
            # h is a list of h_i 
            for h_i in h:
                h_s.append(self.rnn_tanh(self.rnn_d(h_i)))

            output_final = torch.stack(h_s)
#             output_final = output_final.softmax(dim=2)

        # Variante NLP
        if type == 'many-to-many VNLP':
            output_final = output.softmax(dim=2)

        return output_final


def cout_0_1(yhat,y):
    yhat_softmax = yhat.softmax(dim=1)
    pred = torch.argmax(yhat_softmax,dim=1)
    return np.where(pred != y,1,0).sum() / y.size(0)



class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

