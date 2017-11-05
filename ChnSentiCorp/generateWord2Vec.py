from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
import multiprocessing

def word2vec():
    start_time = time.time()
    input_file = './data/seg/all_cut.txt'
    output_model_file = './data/word2vec/word2vec-125.model'
    output_vector_file = './data/word2vec/word2vec-125.vector'

    model = Word2Vec(LineSentence(input_file), size=125, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file, binary=False)

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))

word2vec()