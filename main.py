# gadget_search_engine/main.py

import os
from app.builder import build_and_save_index
from app.predictor import GadgetSearch

def main():
    # Check if the FAISS index exists. If not, build it.
    if not os.path.exists("models/gadgets.faiss"):
        print("Search index not found. Running the builder now...")
        build_and_save_index()
    
    # Initialize the predictor (the search engine)
    try:
        search_engine = GadgetSearch()
    except Exception:
        # Predictor will print a more specific error, so we just exit
        return

    # --- Interactive Search Loop ---
    print("\n--- Gadget Search Engine Ready ---")
    print("Describe the function of a gadget you're looking for (or type 'exit' to quit).")

    while True:
        user_query = input("\n> ")
        if user_query.lower() == 'exit':
            break
        
        # Get search results
        predictions = search_engine.search(user_query, k=3)
        
        print("\n--- Top Results ---")
        if not predictions:
            print("No relevant gadgets found.")
        else:
            for i, pred in enumerate(predictions):
                print(f"{i+1}. Gadget: {pred['gadget_name']} (Similarity: {pred['similarity']:.2f})")
                print(f"   Function: {pred['function']}\n")

if __name__ == '__main__':
    main()
    
    
# ```



# ### ✅ How to Run Your New Project

# 1.  **Install Requirements:**
#     Open your terminal in the `gadget_search_engine` directory.
#     ```bash
#     pip install -r requirements.txt
#     ```

# 2.  **Run the Main Script:**
#     ```bash
#     python main.py
#     ```

#     *   The **first time** you run this, it will see that `models/gadgets.faiss` doesn't exist. It will automatically call the `builder.py` script. You will see a progress bar as it downloads the sentence model and encodes your gadget functions. It will then save the index.
#     *   **Every subsequent time**, it will find the index and load it instantly, skipping the build step.

# 3.  **Test It!**
#     The interactive prompt will start. Try these queries:

#     *   `> I want to fly`
#     *   `> A machine to go to the past or future`
#     *   `> something that helps me understand other languages`
#     *   `> a cannon that shoots air`

#     You will see that the results are now highly relevant, accurate, and returned instantly. This is the correct and modern way to solve your problem.