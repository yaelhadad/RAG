import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load text
with open("frontegg.ai.txt", "r", encoding="utf-8") as f:
    text = f.read()

nltk.download("punkt")
sentences = nltk.sent_tokenize(text)

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# Semantic chunking
chunks = []
current_chunk = [sentences[0]]
chunk_embedding_sum = embeddings[0]
chunk_size = 1

THRESHOLD = 0.5 # similarity threshold for starting a new chunk

for i in range(1, len(sentences)):
    # Compute average embedding of current chunk
    chunk_avg = chunk_embedding_sum / chunk_size
    similarity = cosine_similarity([chunk_avg], [embeddings[i]])[0][0]
    
    if similarity >= THRESHOLD:
        current_chunk.append(sentences[i])
        chunk_embedding_sum += embeddings[i]
        chunk_size += 1
    else:
        chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i]]
        chunk_embedding_sum = embeddings[i]
        chunk_size = 1

# Append last chunk
chunks.append(" ".join(current_chunk))

# Print results
print(f"📊 Created {len(chunks)} semantic chunks\n")
for i, chunk in enumerate(chunks, 1):
    print(f"🔸 Chunk {i}:\n{chunk}\n")
