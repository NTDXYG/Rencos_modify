import sys

import textdistance
from tqdm import tqdm

def retriever(file_dir):

    with open(file_dir + "/train/train.code.src", 'r') as fso,  open(file_dir + "/train/train.nl.tgt", 'r') as fsu, \
            open(file_dir + "/train/train.ast.src", 'r') as ast_list:
        sources = [line.strip() for line in fso.readlines()]
        asts = [line.strip() for line in ast_list.readlines()]
        summaries = [line.strip() for line in fsu.readlines()]
    with open(file_dir+"/test/test.ast.src") as ft, open(file_dir+"/test/test.ref.src.0", 'w') as fwo, \
            open(file_dir+"/output/ast.out", 'w') as fws:
        queries = [line.strip() for line in ft.readlines()]

        for i in tqdm(range(len(queries))):
            max_score = 0
            max_idx = 0

            for j in range(len(asts)):
                score = textdistance.levenshtein.normalized_similarity(queries[i], asts[j])
                if(score>max_score):
                    max_score = score
                    max_idx = j

            fwo.write(sources[max_idx] + '\n')
            fws.write(summaries[max_idx] + '\n')


if __name__ == '__main__':
    root = 'samples/%s'%sys.argv[1]
    retriever(root)
