import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    return embeddings

def visualize_embeddings(embeddings, words_to_visualize):
    words = []
    vectors = []
    for word in words_to_visualize:
        if word in embeddings:
            words.append(word)
            vectors.append(embeddings[word])

    vectors = np.array(vectors)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
    plt.show()

if __name__ == "__main__":
    embeddings = load_embeddings("model/vectors.txt")
    words_to_visualize = ["книга", "автор", "читатель", "библиотека", "роман", "поэзия"]
    visualize_embeddings(embeddings, words_to_visualize)