import torch
import re
import numpy as np
import selfies as sf


_atoms = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar',
          'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Cu', 'Ga', 'Ge', 'As', 'Se',
          'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
          'Pd', 'Ag', 'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr',
          'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er', 'Tm', 'Yb',
          'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
          'Bi', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Pu', 'Am', 'Cm',
          'Bk', 'Cf', 'Es', 'Fm', 'Md', 'Lr', 'Rf', 'Db', 'Sg', 'Mt',
          'Ds', 'Rg', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


def get_tokenizer_re(atoms):
    return re.compile('('+'|'.join(atoms)+r'|\%\d\d|.)')


_atoms_re = get_tokenizer_re(_atoms)


__i2t = {
    0: 'unused', 1: '>', 2: '<', 3: '2', 4: 'F', 5: 'Cl', 6: 'N',
    7: '[', 8: '6', 9: 'O', 10: 'c', 11: ']', 12: '#',
    13: '=', 14: '3', 15: ')', 16: '4', 17: '-', 18: 'n',
    19: 'o', 20: '5', 21: 'H', 22: '(', 23: 'C',
    24: '1', 25: 'S', 26: 's', 27: 'Br'#,28:'B'#,28: '/', 29: '@', 30: '7', 31: '\\',32: '.'
}


__t2i = {
    '>': 1, '<': 2, '2': 3, 'F': 4, 'Cl': 5, 'N': 6, '[': 7, '6': 8,
    'O': 9, 'c': 10, ']': 11, '#': 12, '=': 13, '3': 14, ')': 15,
    '4': 16, '-': 17, 'n': 18, 'o': 19, '5': 20, 'H': 21, '(': 22,
    'C': 23, '1': 24, 'S': 25, 's': 26, 'Br': 27#,'I': 27, 'B':28 #, '/': 28, '@': 29,'7': 30,
    #'\\': 31, '.': 32
}

__symbol_to_idx = np.load("s2i.npy",allow_pickle=True).item()

__idx_to_symbol = np.load("i2s.npy",allow_pickle=True).item()


# def smiles_tokenizer(line, atoms=None):
#     """
#     Tokenizes SMILES string atom-wise using regular expressions. While this
#     method is fast, it may lead to some mistakes: Sn may be considered as Tin
#     or as Sulfur with Nitrogen in aromatic cycle. Because of this, you should
#     specify a set of two-letter atoms explicitly.

#     Parameters:
#          atoms: set of two-letter atoms for tokenization
#     """
#     if atoms is not None:
#         reg = get_tokenizer_re(atoms)
#     else:
#         reg = _atoms_re
#     return reg.split(line)[1::2]


# def encode(sm_list, pad_size=50):
#     """
#     Encoder list of smiles to tensor of tokens
#     """
#     res = []
#     lens = []
#     for s in sm_list:
#         tokens = ([1] + [__t2i[tok]
#                   for tok in smiles_tokenizer(s)])[:pad_size - 1]
#         lens.append(len(tokens))
#         tokens += (pad_size - len(tokens)) * [2]
#         res.append(tokens)

#     return torch.tensor(res).long(), lens



# def decode(tokens_tensor):
#     """
#     Decodes from tensor of tokens to list of smiles
#     """

#     smiles_res = []

#     for i in range(tokens_tensor.shape[0]):
#         cur_sm = ''
#         for t in tokens_tensor[i].detach().cpu().numpy():
#             if t == 2:
#                 break
#             elif t > 2:
#                 cur_sm += __i2t[t]

#         smiles_res.append(cur_sm)

#     return smiles_res



# def get_vocab_size():
#     return len(__i2t)

def get_vocab_size():
    return len(__idx_to_symbol)

def encode(selfies_list,pad_size=105):
    """
    Encoder list of smiles to tensor of tokens
    """
    res = []
    lens = []
    for s in selfies_list: 
        label = ([1]+sf.selfies_to_encoding(
           selfies=s,
           vocab_stoi=__symbol_to_idx,
           pad_to_len=pad_size,
           enc_type="label"
        ))[:pad_size-1]
        lens.append(len(label))
        label += (pad_size - len(label)) * [2]
        res.append(label)
        

    return torch.tensor(res).long(), lens

def decode(tokens_tensor):
    """
    Decodes from tensor of tokens to list of smiles
    """

    selfies_res = []

    for i in range(tokens_tensor.shape[0]):
        cur_sm = ''
        for t in tokens_tensor[i].detach().cpu().numpy():
            if t == 2:
                break
            elif t>2:
                cur_sm += __idx_to_symbol[t]
#         cur_sm = sf.encoding_to_selfies(
#             encoding = tokens_tensor[i].detach().cpu().numpy(),
#             vocab_itos = __idx_to_symbol,
#             enc_type = 'label'
#         )
        selfies_res.append(cur_sm)

    return selfies_res
