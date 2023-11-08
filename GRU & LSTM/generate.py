from textloader import  string2code, id2lettre
import torch
from utils import *
from torch.utils.data import Subset
from torch.utils.data import Dataset,DataLoader
from textloader import pad_collate_fn


#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(model, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    # Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    if start == '' : start = ' '
    sentence = start
    # format ( batch_size , length_seq) --> ( 1 , length_seq)
    start_emb = model.embedder(string2code(start).view(-1,1))

    if isinstance(model.rnn,LSTM):
        h_i , _ , c_i , _ = model.rnn(start_emb)
    elif isinstance(model.rnn,GRU):
        h_i , _  = model.rnn(start_emb)

    probas = model.rnn.decode(h_i,type='many-to-one').softmax(dim=1)

    multinomial_dist = Multinomial(1, probs=probas)
    # get a binary vector 
    sample = multinomial_dist.sample()
    sentence += code2string([torch.argmax(sample).item()])
    x_i = model.embedder(string2code(sentence[-1]))

    for nb_cara_generated in range(maxlen+1):
        if isinstance(model.rnn,LSTM):
            h_i , c_i = model.rnn.one_step(x_i,h_i,c_i)
        elif isinstance(model.rnn,GRU):
            h_i  = model.rnn.one_step(x_i,h_i)
        
        probas = model.rnn.decode(h_i,type='many-to-one').softmax(dim=1)
        multinomial_dist = Multinomial(1, probs=probas)
        sample = multinomial_dist.sample()
        cara_to_add =code2string([torch.argmax(sample).item()])
        sentence += cara_to_add
        x_i = model.embedder(string2code(sentence[-1]))
        if cara_to_add == id2lettre[eos]:
            break

    print(f"Generation of {nb_cara_generated} caracteres from <{start}> ")
    return sentence



def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute
