# Gadget Semantic Search Engine

A high-performance semantic search engine designed to find gadgets based on natural language descriptions of their functions. This project leverages sentence embeddings and a high-speed vector search library to understand the *meaning* behind a user's query, providing highly relevant results even when keywords don't match exactly.

## 🚀 Key Features

-   **🧠 Semantic Search**: Powered by `Sentence-Transformers`, the engine understands context and meaning, not just keywords.
-   **⚡ Blazing Fast**: Utilizes `FAISS` (Facebook AI Similarity Search) to query thousands of documents in milliseconds.
-   **💡 Data Augmentation**: Includes a script to automatically enrich the dataset by creating function variations with synonyms using `NLTK` and `WordNet`, overcoming the vocabulary mismatch problem.
-   **📦 Modular & Robust**: The codebase is organized into logical components for data augmentation, index building, and prediction.
-   **💬 Interactive**: Comes with a simple command-line interface to test and interact with the search engine directly.

## 🛠️ Technology Stack

-   **Python 3.10+**
-   **Core Libraries**:
    -   [Sentence-Transformers](https://sbert.net/): For generating high-quality sentence embeddings.
    -   [Faiss (Facebook AI Similarity Search)](https://faiss.ai/): For efficient similarity search in vector space.
    -   [Pandas](https://pandas.pydata.org/): For data manipulation.
    -   [NLTK](https://www.nltk.org/): For data augmentation via synonym replacement.
    -   [NumPy](https://numpy.org/): For numerical operations.

## 📁 Project Structure

```
gadget_search_engine/
│
├── app/
│   ├── __init__.py
│   ├── augmenter.py       # Script to augment the dataset with synonyms
│   ├── builder.py         # Script to build the FAISS search index
│   └── predictor.py       # Core search logic class
│
├── data/
│   └── gadgets.csv        # Your initial raw dataset of gadgets and functions
│
├── models/
│   └── .gitkeep           # Generated models (FAISS index, etc.) will be stored here
│
├── main.py                # Main entry point to run the interactive search
├── requirements.txt       # Project dependencies
└── README.md              # You are here!
```

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd gadget_search_engine
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

-   **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
-   **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

### 3. Install Dependencies

Install all the required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

##  workflow-and-usage'
## 📋 Workflow and Usage

The project follows a simple three-step workflow. Run these commands from the root of the `gadget_search_engine` directory.

### Step 1: Prepare Your Data

Ensure your raw data is in `data/gadgets.csv` with two columns: `gadget_name` and `function`.

```csv
gadget_name,function
Anywhere Door,Teleports users to any desired location instantly
Hopter,A small propeller device for personal flight that attaches to the body
...```

### Step 2: Augment the Dataset (Recommended)

This step will read your `gadgets.csv`, create new variations of the functions using synonyms, and save the result to `data/gadgets_augmented.csv`. This significantly improves search relevance.

```bash
python app/augmenter.py
```

This will also trigger a one-time download of necessary `NLTK` data packages.

### Step 3: Build the Search Index

This script processes the augmented data, generates vector embeddings, and builds the high-speed FAISS index. The final artifacts are saved in the `models/` directory.

```bash
python app/builder.py
```*(Note: If you skip this step, `main.py` will automatically try to run it, but it's good practice to do it manually first.)*

### Step 4: Run the Interactive Search!

Start the application and begin searching for gadgets.

```bash
python main.py
```

You will be greeted with an interactive prompt where you can type your queries.

```
--- Gadget Search Engine Ready ---
Describe the function of a gadget you're looking for (or type 'exit' to quit).

> I want to travel to the past
```

## 🔬 How It Works

1.  **Data Augmentation**: The `augmenter.py` script uses NLTK's WordNet corpus to find nouns and verbs in each function description and replaces them with synonyms, creating a richer, more diverse dataset.

2.  **Indexing**: The `builder.py` script takes each function (original and augmented), and the `SentenceTransformer` model (`all-MiniLM-L6-v2`) converts it into a 384-dimensional numerical vector (embedding). These vectors are then loaded into a `FAISS` index, which organizes them in a way that allows for extremely fast nearest-neighbor search.

3.  **Searching**: When you enter a query, the `predictor.py` logic encodes your query into a vector using the same model. It then asks the FAISS index for the vectors that are mathematically closest (highest cosine similarity) to your query's vector. Finally, it de-duplicates the results to ensure each gadget appears only once, represented by its best-scoring function.


