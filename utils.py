import faiss
import numpy as np

def reconstruct_binary_vector_from_index(index_path):
    findex = faiss.read_index_binary(index_path)
    vectors = []
    for i in range(findex.ntotal):
        vectors.append(findex.reconstruct(i))
    vectors = np.array(vectors)
    return vectors
