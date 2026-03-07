import re
try:
    from underthesea import word_tokenize
except ImportError:
    print("Warning: underthesea not installed. Using basic split.")
    word_tokenize = lambda x: x.split()

def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase.
    2. Removing special characters.
    3. Tokenizing (using underthesea or basic split).
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters but keep some punctuation for sentiment/intent
    # Keep Vietnamese characters, digits, and !? marks
    text = re.sub(r'[^\w\s!?]', ' ', text)
    # Add spaces around ! and ? to make them separate tokens
    text = re.sub(r'([!?])', r' \1 ', text)
    # Replace multiple spaces with one
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Tokenize
    tokens = word_tokenize(text, format="text")
    
    return tokens

if __name__ == "__main__":
    sample = "Tin nóng: Người ngoài hành tinh xuất hiện tại Hà Nội!"
    print(f"Original: {sample}")
    print(f"Cleaned: {clean_text(sample)}")