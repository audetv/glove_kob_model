import numpy as np

def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    return embeddings

def word_analogy(embeddings, word_a, word_b, word_c, top_n=5):
    if word_a not in embeddings or word_b not in embeddings or word_c not in embeddings:
        return []

    vec_a = embeddings[word_a]
    vec_b = embeddings[word_b]
    vec_c = embeddings[word_c]
    target_vector = vec_b - vec_a + vec_c

    similarities = {}
    for word, vector in embeddings.items():
        if word in [word_a, word_b, word_c]:
            continue
        similarity = np.dot(target_vector, vector) / (np.linalg.norm(target_vector) * np.linalg.norm(vector))
        similarities[word] = similarity

    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]

if __name__ == "__main__":
    embeddings = load_embeddings("model/vectors.txt")
    word_a, word_b, word_c = "король", "мужчина", "женщина"
    result = word_analogy(embeddings, word_a, word_b, word_c, top_n=5)
    print(f"Аналогия: {word_a} - {word_b} + {word_c} = ?")
    for word, similarity in result:
        print(f"{word}: {similarity:.4f}")