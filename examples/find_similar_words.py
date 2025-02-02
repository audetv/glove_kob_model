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

def find_similar_words(embeddings, target_word, top_n=5):
    if target_word not in embeddings:
        return []

    target_vector = embeddings[target_word]
    similarities = {}

    for word, vector in embeddings.items():
        if word == target_word:
            continue
        similarity = np.dot(target_vector, vector) / (np.linalg.norm(target_vector) * np.linalg.norm(vector))
        similarities[word] = similarity

    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]

if __name__ == "__main__":
    embeddings = load_embeddings("model/vectors.txt")
    target_word = "предиктор"
    similar_words = find_similar_words(embeddings, target_word, top_n=10)
    print(f"Слова, близкие к '{target_word}':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")