from tqdm import tqdm
from model.tokenizer import _punctuations
from cleaners import english_cleaners

if __name__ == "__main__":
    with open("preprocess/metadata_pp_graphem.csv", 'w') as o:
        with open('preprocess/metadata.csv') as f:
            for row in tqdm(f):
                t = row.strip().split("|")
                text = english_cleaners(t[2])
                o.write(t[0] + '|' + text + '\n')
