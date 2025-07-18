# # gadget_search_engine/app/builder.py

# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import os

# def build_and_save_index():
#     """
#     1. Loads gadget data from CSV.
#     2. Encodes gadget functions into vector embeddings using a sentence transformer.
#     3. Normalizes embeddings for cosine similarity search.
#     4. Builds a FAISS index for efficient searching.
#     5. Saves the index, the model, and the dataframe for later use.
#     """
#     print("Starting index build process...")
    
#     # Create the 'models' directory if it doesn't exist
#     if not os.path.exists("models"):
#         os.makedirs("models")

#     # --- 1. Load Data ---
#     try:
#         df = pd.read_csv("data/gadgets.csv")
#         df.dropna(subset=['gadget_name', 'function'], inplace=True)
#         # Reset index to ensure it's a clean 0-based sequence
#         df.reset_index(drop=True, inplace=True)
#     except FileNotFoundError:
#         print("Error: `data/gadgets.csv` not found. Please create it first.")
#         return

#     print(f"Loaded {len(df)} gadget functions.")

#     # --- 2. Encode Functions to Embeddings ---
#     print("Loading sentence transformer model and encoding functions...")
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(df['function'].tolist(), show_progress_bar=True)
    
#     # Get the dimension of the embeddings (e.g., 384 for all-MiniLM-L6-v2)
#     d = embeddings.shape[1]

#     # --- 3. Normalize Embeddings for Cosine Similarity ---
#     # FAISS uses L2 distance for search. For normalized vectors, L2 distance
#     # is directly related to cosine similarity, but faster to compute.
#     # cos_sim(A, B) = A . B
#     # We will use IndexFlatIP (Inner Product) which is equivalent to cosine similarity for normalized vectors.
#     faiss.normalize_L2(embeddings)

#     # --- 4. Build FAISS Index ---
#     print("Building FAISS index...")
#     # Using IndexFlatIP for maximum inner product search (cosine similarity)
#     index = faiss.IndexFlatIP(d)
    
#     # We use an ID map to relate the index's internal IDs back to our dataframe's original indices
#     index = faiss.IndexIDMap(index)
    
#     # Add vectors to the index with their corresponding dataframe indices
#     index.add_with_ids(embeddings, df.index.values)

#     # --- 5. Save Artifacts ---
#     print("Saving index and data...")
#     faiss.write_index(index, "models/gadgets.faiss")
#     df.to_csv("models/gadget_data.csv", index=False)
    
#     print("\n✅ Index build complete. Artifacts saved in 'models/' directory.")

# if __name__ == '__main__':
#     build_and_save_index()





# gadget_search_engine/app/builder.py

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def build_and_save_index():
    """
    1. Loads augmented gadget data from CSV.
    2. Encodes gadget functions into vector embeddings using a sentence transformer.
    3. Normalizes embeddings for cosine similarity search.
    4. Builds a FAISS index for efficient searching.
    5. Saves the index and the corresponding data for later use.
    """
    print("Starting index build process...")

    # Create the 'models' directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # --- 1. Load Augmented Data ---
    # This is the key change: we load the augmented file.
    augmented_data_path = "data/gadgets_augmented.csv"
    try:
        df = pd.read_csv(augmented_data_path)
        df.dropna(subset=['gadget_name', 'function'], inplace=True)
        # Reset index to ensure it's a clean 0-based sequence, which is crucial for faiss.IndexIDMap
        df.reset_index(drop=True, inplace=True)
    except FileNotFoundError:
        print(f"❌ Error: Augmented data file not found at '{augmented_data_path}'.")
        print("➡️ Please run the data augmentation script first by executing:")
        print("   python app/augmenter.py")
        return

    print(f"Loaded {len(df)} augmented gadget functions.")

    # --- 2. Encode Functions to Embeddings ---
    print("Loading sentence transformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding functions into vector embeddings. This may take a while for large datasets...")
    embeddings = model.encode(df['function'].tolist(), show_progress_bar=True)

    # Get the dimension of the embeddings (e.g., 384 for all-MiniLM-L6-v2)
    d = embeddings.shape[1]
    print(f"Embeddings created with dimension: {d}")

    # --- 3. Normalize Embeddings for Cosine Similarity ---
    # FAISS search with IndexFlatIP is based on inner product.
    # For normalized vectors, inner product is equivalent to cosine similarity.
    # This step is crucial for getting meaningful similarity scores.
    faiss.normalize_L2(embeddings)

    # --- 4. Build FAISS Index ---
    print("Building FAISS index...")
    # Using IndexFlatIP for maximum inner product search (our cosine similarity)
    index = faiss.IndexFlatIP(d)

    # We use an ID map to relate the index's internal IDs back to our dataframe's original indices
    # This allows us to retrieve the original text data after a search
    index = faiss.IndexIDMap(index)

    # Add vectors to the index with their corresponding dataframe indices
    index.add_with_ids(embeddings, df.index.values.astype('int64'))

    print(f"FAISS index built successfully. Total entries: {index.ntotal}")

    # --- 5. Save Artifacts ---
    faiss_index_path = "models/gadgets.faiss"
    data_path = "models/gadget_data.csv"

    print(f"Saving FAISS index to '{faiss_index_path}'...")
    faiss.write_index(index, faiss_index_path)

    print(f"Saving corresponding gadget data to '{data_path}'...")
    df.to_csv(data_path, index=False)

    print("\n✅ Index build complete. All artifacts saved in 'models/' directory.")

if __name__ == '__main__':
    # This allows the script to be run directly from the command line
    build_and_save_index()
    
    
    
# ```

# ### Summary of Changes and Logic

# *   **Reads Augmented Data:** It explicitly looks for `data/gadgets_augmented.csv`.
# *   **Clear Error Message:** If the file isn't found, it gives you the exact command to run (`python app/augmenter.py`) to generate it. This makes the workflow very user-friendly.
# *   **Vector Normalization:** It includes the critical `faiss.normalize_L2(embeddings)` step, which ensures that the search correctly calculates cosine similarity.
# *   **ID Mapping:** It correctly uses `faiss.IndexIDMap` and `df.index.values` to create a link between the vectors in the search index and the original rows in the `gadget_data.csv` file.
# *   **Saves Data:** It saves the augmented dataframe to `models/gadget_data.csv`. This is essential because the `predictor` needs this file to look up the gadget name from an ID returned by the search.