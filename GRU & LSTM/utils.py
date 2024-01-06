import torch
import torch.nn.functional as F


def get_y(x):
    y = x[1:]
    zero_row = torch.zeros(1, x.size(1), dtype=x.dtype)  # Crée un tensor de zéros avec la même dimension que x
    return torch.cat((y,zero_row), dim=0).to(torch.long)
    

def maskedCrossEntropy(output, target, padcar=0,reduce='mean'):
    """
    :param output: Tenseur length x batch x output_dim
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    # Créer un masque pour les caractères de padding
    mask = (target != padcar).float()

    # Appliquer la fonction de perte en utilisant le masque
    loss = F.cross_entropy(output.transpose(1, 2), target, reduction='none') * mask

    # Appliquer la réduction spécifiée
    if reduce == 'mean':
        loss = loss.mean()
    elif reduce == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("L'argument 'reduce' doit être 'mean' ou 'sum'.")

    return loss