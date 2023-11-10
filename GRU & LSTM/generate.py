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

def get_topK(log_probas,k):
    topk_values = log_probas.topk(k,dim=1)[0][0]
    topk_indices = log_probas.topk(k,dim=1)[1][0]

    return topk_values,topk_indices

'''def generate_beam(model, eos, k, start="", maxlen=200):
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
    if start == '' : start = ' '
    sentence = start
    sentences = [[sentence] * k][0]
    # format ( batch_size , length_seq) --> ( 1 , length_seq)
    start_emb = model.embedder(string2code(start).view(-1,1))
    tree =  dict()

    if isinstance(model.rnn,LSTM):
        h_i , _ , c_i , _ = model.rnn(start_emb)
    elif isinstance(model.rnn,GRU):
        h_i , _  = model.rnn(start_emb)

    probas = model.rnn.decode(h_i,type='many-to-one').softmax(dim=1)
    log_probas = -torch.log(probas)
    topk_values,topk_indices = get_topK(log_probas,k)
    liste_x_i = []
    for i in range(k):
        sentences[i] +=code2string([topk_indices[i].item()])
        tree[i] = (sentences[i] , topk_values[i].item() , dict())
        liste_x_i.append(model.embedder(string2code(sentences[i][-1])))
    print("iteration 0 ",tree)
    print()

    breaked = False
    for nb_cara_generated in range(maxlen+1):
        for i in range(k):
            if isinstance(model.rnn,LSTM):
                h_i , c_i = model.rnn.one_step(liste_x_i[i],h_i,c_i)
            elif isinstance(model.rnn,GRU):
                h_i  = model.rnn.one_step(liste_x_i[i],h_i)
            
            probas = model.rnn.decode(h_i,type='many-to-one').softmax(dim=1)
            log_probas = -torch.log(probas)
            topk_values,topk_indices = get_topK(log_probas,k)
            liste_x_i = []
            sentences = [[sentences[i]] * k][0]
            
            for j in range(k):
                sentences[j] += code2string([topk_indices[j].item()])
                tree[i][2][j] = (sentences[j] , topk_values[j].item(),dict())
                liste_x_i.append(model.embedder(string2code(sentences[i][-1])))
                if code2string([topk_indices[j].item()]) == id2lettre[eos]:
                    breaked = True
                    break
                else:continue
            if breaked:
                break
            else:continue
        if breaked:break
        else:continue
            
    print(f"iteration {nb_cara_generated+1}",tree)


    return tree'''


def generate_beam(model, eos, k, start="", maxlen=200):
    if start == '':
        start = ' '

    best_seq = 0
    print("<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>")
    print("Beam Search generation starting ")

    # Initialize the beam with the first sequence
    beam = [{'sequence': start, 'log_prob': 0.0, 'prev_index': None}]
    
    # Format (batch_size, length_seq) --> (1, length_seq)
    start_emb = model.embedder(string2code(start).view(-1, 1))
    
    if isinstance(model.rnn, LSTM):
        h_i,_, c_i,_ = model.rnn(start_emb)
    elif isinstance(model.rnn, GRU):
        h_i, _= model.rnn(start_emb)

    best_seq = None
    
    for nb_cara_generated in range(maxlen):
        new_beam = []
        end_flag = True
        
        for i, candidate in enumerate(beam):
            # Extract information from the candidate
            sequence = candidate['sequence']
            log_prob = candidate['log_prob']
            prev_index = i
            
            if isinstance(model.rnn, LSTM):
                h_i,c_i = model.rnn.one_step(model.embedder(string2code(sequence[-1])), h_i, c_i)
            elif isinstance(model.rnn, GRU):
                h_i = model.rnn.one_step(model.embedder(string2code(sequence[-1])), h_i)
            
            probas = model.rnn.decode(h_i, type='many-to-one').softmax(dim=1)
            log_probas = -torch.log(probas)
            
            # Get the top K values and indices
            topk_values, topk_indices = get_topK(log_probas, k)
            
            for j in range(k):
                new_seq = sequence + code2string([topk_indices[j].item()])
                new_log_prob = log_prob + topk_values[j].item()
                
                new_beam.append({'sequence': new_seq, 'log_prob': new_log_prob, 'prev_index': prev_index})
                
                # Check for end-of-sequence symbol
                if code2string([topk_indices[j].item()]) == id2lettre[eos]:
                    end_flag = False
                    best_seq = i
                    break
                else:continue
            if not end_flag:break
            else:continue
        
        # Sort the new beam based on log probabilities
        new_beam = sorted(new_beam, key=lambda x: x['log_prob'], reverse=True)[:k]
        
        # Break if end-of-sequence symbol is found in any of the new sequences
        if not end_flag:
            break
        
        # Update the beam for the next iteration
        beam = new_beam


    print(f"Generation of {nb_cara_generated} caracteres from <{start}> ")
    print("Best sequence : ",beam[best_seq]['sequence'])
    print("<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>\n")

    
    return beam

def nucleus_sampling(probas, alpha):
    sorted_indices = torch.argsort(probas, descending=True)
    cumulative_probs = torch.cumsum(probas[0,sorted_indices], dim=-1)
    
    # Find the smallest index whose cumulative probability exceeds alpha
    nucleus_index = torch.where(cumulative_probs >= alpha)[0][0]

    # Extract the symbols that exceed the nucleus probability
    nucleus_symbols = sorted_indices[:nucleus_index + 1]
    
    return nucleus_symbols

def generate_beam_nucleus(model, eos, k, alpha, start="", maxlen=200):
    if start == '':
        start = ' '
    print("<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>")   
    print("Nucleus Beam Search generation starting ")


    best_seq = 0

    # Initialize the beam with the first sequence
    beam = [{'sequence': start, 'log_prob': 0.0, 'prev_index': None}]
    
    # Format (batch_size, length_seq) --> (1, length_seq)
    start_emb = model.embedder(string2code(start).view(-1, 1))
    
    if isinstance(model.rnn, LSTM):
        h_i, _,c_i,_ = model.rnn(start_emb)
    elif isinstance(model.rnn, GRU):
        h_i,_ = model.rnn(start_emb)
    
    for nb_cara_generated in range(maxlen):
        new_beam = []
        end_flag = True
        
        for i, candidate in enumerate(beam):
            # Extract information from the candidate
            sequence = candidate['sequence']
            log_prob = candidate['log_prob']
            prev_index = i
            
            if isinstance(model.rnn, LSTM):
                h_i, c_i = model.rnn.one_step(model.embedder(string2code(sequence[-1])), h_i, c_i)
            elif isinstance(model.rnn, GRU):
                h_i = model.rnn.one_step(model.embedder(string2code(sequence[-1])), h_i)
            
            probas = model.rnn.decode(h_i, type='many-to-one').softmax(dim=1)
            nucleus_indices = nucleus_sampling(probas, alpha)
            for j in nucleus_indices[0]:
                new_seq = sequence + code2string([j.item()])
                new_log_prob = log_prob - torch.log(probas[0,j]).item()
                
                new_beam.append({'sequence': new_seq, 'log_prob': new_log_prob, 'prev_index': prev_index})
                
                # Check for end-of-sequence symbol
                if j.item() == eos:
                    end_flag = False
                    best_seq = i
                    break
 
                else:continue
            if not end_flag:break
            else:continue
        
        # Sort the new beam based on log probabilities
        new_beam = sorted(new_beam, key=lambda x: x['log_prob'], reverse=True)[:k]
        
        # Break if end-of-sequence symbol is found in any of the new sequences
        if not end_flag:
            break
        
        # Update the beam for the next iteration
        beam = new_beam
    
    print(f"Generation of {nb_cara_generated} caracteres from <{start}> ")
    print("Best sequence : ",beam[best_seq]['sequence'])
    print("<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>\n")

    return beam



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
