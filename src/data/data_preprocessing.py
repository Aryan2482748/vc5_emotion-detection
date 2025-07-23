import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    except Exception as e:
        logging.error(f"Lemmatization failed: {e}")
        return text

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        filtered = [word for word in str(text).split() if word not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        logging.error(f"Removing stop words failed: {e}")
        return text

def removing_numbers(text: str) -> str:
    """Remove all digits from the text."""
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logging.error(f"Removing numbers failed: {e}")
        return text

def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logging.error(f"Lowercasing failed: {e}")
        return text

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the text."""
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logging.error(f"Removing punctuations failed: {e}")
        return text

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"Removing URLs failed: {e}")
        return text

def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set text to NaN if sentence has fewer than 3 words.
    Modifies the DataFrame in place and returns it.
    """
    try:
        mask = df['text'].apply(lambda x: len(str(x).split()) < 3)
        df.loc[mask, 'text'] = np.nan
        logging.info("Removed small sentences with fewer than 3 words.")
        return df
    except Exception as e:
        logging.error(f"Removing small sentences failed: {e}")
        return df

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the 'content' column of the DataFrame.
    """
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("Normalized text in DataFrame.")
        return df
    except Exception as e:
        logging.error(f"Normalizing text failed: {e}")
        return df

def normalized_sentence(sentence: str) -> str:
    """
    Apply all preprocessing steps to a single sentence.
    """
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Normalizing sentence failed: {e}")
        return sentence

def main() -> None:
    """Main function to orchestrate data preprocessing."""
    try:
        # Load raw train and test data
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
        logging.info("Loaded raw train and test data.")

        # Ensure the text column is named 'content' for processing
        if 'text' in train_data.columns:
            train_data = train_data.rename(columns={'text': 'content'})
        if 'text' in test_data.columns:
            test_data = test_data.rename(columns={'text': 'content'})

        # Normalize train and test data
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)

        # Save processed data to CSV files
        os.makedirs("data/processed", exist_ok=True)
        train_data.to_csv("data/processed/train.csv", index=False)
        test_data.to_csv("data/processed/test.csv", index=False)
        logging.info("Saved processed train and test data.")
    except Exception as e:
        logging.critical(f"Data preprocessing pipeline failed: {e}")

if __name__ == "__main__":
    main()