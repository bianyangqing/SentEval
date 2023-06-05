# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Clone GenSen repo here: https://github.com/Maluuba/gensen.git
And follow instructions for loading the model used in batcher
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import logging
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import sentence_transformers
import torch
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_NAME = "GanymedeNil/text2vec-large-chinese"

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, )
embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name,
                                                                   device=DEVICE)
# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval






# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    # Tokenize the input texts
    result = []
    for sent in batch:
        result.append(embeddings.embed_query(sent))
    result = np.vstack(result)
    return result


# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS14']
    results = se.eval(transfer_tasks)
    print(results)
