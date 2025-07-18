# gadget_search_engine/app/augmenter.py

import pandas as pd
import nltk
from nltk.corpus import wordnet
import random

# --- This try-except block has been updated to be fully comprehensive for this script ---
try:
    # Check if all required NLTK data, including specific sub-packages, is available
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    # --- START OF FIX ---
    # Added a check for the specific English tagger data
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    # --- END OF FIX ---
    nltk.data.find('corpora/wordnet')

except LookupError:
    # If any of the resources are not found, download them all to be safe.
    print("Downloading necessary NLTK data packages...")
    packages = [
        'punkt', 'punkt_tab', 'averaged_perceptron_tagger',
        # --- START OF FIX ---
        # Added the missing package to the download list
        'averaged_perceptron_tagger_eng',
        # --- END OF FIX ---
        'wordnet'
    ]
    for package in packages:
        print(f'   > Downloading {package}...')
        nltk.download(package, quiet=True)
    print("✅ All NLTK data downloaded.")


def get_synonyms(word, pos_tag):
    """
    Get synonyms for a word given its part-of-speech tag.
    (e.g., 'v' for verb, 'n' for noun)
    """
    # Map NLTK's detailed POS tags to WordNet's simpler tags
    if pos_tag.startswith('J'):
        wn_pos = wordnet.ADJ
    elif pos_tag.startswith('V'):
        wn_pos = wordnet.VERB
    elif pos_tag.startswith('N'):
        wn_pos = wordnet.NOUN
    elif pos_tag.startswith('R'):
        wn_pos = wordnet.ADV
    else:
        wn_pos = None

    if not wn_pos:
        return []

    synsets = wordnet.synsets(word, pos=wn_pos)
    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def augment_sentence(sentence, num_augmentations=3):
    """
    Creates variations of a sentence by replacing important words with synonyms.
    """
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    augmented_sentences = {sentence}
    replaceable_words = []
    for word, tag in pos_tags:
        synonyms = get_synonyms(word, tag)
        if synonyms:
            replaceable_words.append((word, synonyms))

    if not replaceable_words:
        return [sentence]

    # Generate more attempts than needed to ensure we get unique augmentations
    for _ in range(num_augmentations * 10): # Increased attempts for better variety
        if len(augmented_sentences) > num_augmentations:
            break

        new_words = list(words)
        try:
            word_to_replace, synonyms = random.choice(replaceable_words)
            new_synonym = random.choice(synonyms)
        except IndexError:
            continue

        # Replace all occurrences of the word
        for i, word in enumerate(new_words):
            if word == word_to_replace:
                new_words[i] = new_synonym

        augmented_sentences.add(" ".join(new_words))

    return list(augmented_sentences)


def run_augmentation(input_file, output_file):
    """
    Reads a CSV, augments the 'function' column, and saves to a new CSV.
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    augmented_rows = []

    print("Augmenting data... this may take a moment.")
    for index, row in df.iterrows():
        gadget_name = row['gadget_name']
        original_function = row['function']

        if not isinstance(original_function, str):
            continue

        augmented_functions = augment_sentence(original_function, num_augmentations=4)

        for func in augmented_functions:
            augmented_rows.append({'gadget_name': gadget_name, 'function': func})

    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.to_csv(output_file, index=False)

    print(f"\nAugmentation complete. Original: {len(df)} rows. Augmented: {len(augmented_df)} rows.")
    print(f"✅ Saved augmented data to {output_file}")


if __name__ == '__main__':
    run_augmentation(input_file="data/gadgets.csv", output_file="data/gadgets_augmented.csv")