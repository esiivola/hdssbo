import nltk
import numpy as np
import six
import pdb


gram="""smiles -> branched_atom
smiles -> smiles branched_atom
atom -> 'C'
atom -> 'N'
atom -> 'O'
atom -> 'F'
branched_atom -> atom
branched_atom -> atom '1'
branched_atom -> atom '(' smiles ')'
Nothing -> None"""

def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''

def get_zinc_tokenizer_(cfg):
    long_tokens = [l for l in filter(lambda a: len(a) > 1, cfg._lexical_index.keys())]
    replacements = ['$','%','^'] # ,'&']
    assert len(long_tokens) == len(replacements)
    for token in replacements: 
        assert not token in cfg._lexical_index
    
    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens
    
    return tokenize

def get_zinc_tokenizer(cfg):
    long_tokens = [l for l in filter(lambda a: len(a) > 1, cfg._lexical_index.keys())]    
    def tokenize(smiles):
        tokens = []
        for token in smiles:
            tokens.append(token)
        return tokens
    
    return tokenize


def to_smiles(one_hot):
    productions = get_GCFG().productions()
    prod_seq = [[productions[one_hot[index,t].argmax()] 
                    for t in range(one_hot.shape[1])] 
                for index in range(one_hot.shape[0])]
    return [prods_to_eq(prods) for prods in prod_seq]

def to_one_hot(smiles, MAX_LEN, NCHARS):
    """ Encode a list of smiles strings to one-hot vectors """
    GCFG = get_GCFG()
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(GCFG.productions()):
        prod_map[prod] = ix
    #tokenize = get_zinc_tokenizer(GCFG)
    tokens = smiles # map(tokenize, smiles)
    parser = nltk.ChartParser(GCFG)
    parse_trees = [next(parser.parse(t)) for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((0, MAX_LEN, NCHARS), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        if num_productions <= MAX_LEN:
            one_hot_i = np.zeros((1, MAX_LEN, NCHARS), dtype=np.float32)
            one_hot_i[0, np.arange(num_productions),indices[i]] = 1.
            one_hot_i[0, np.arange(num_productions, MAX_LEN),-1] = 1.
            one_hot = np.concatenate((one_hot, one_hot_i))
    return one_hot

def get_GCFG():
    # form the CFG and get the start symbol
    GCFG = nltk.CFG.fromstring(gram)
    return GCFG