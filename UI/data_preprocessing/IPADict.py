import pandas as pd
import re


class IpaDictionary:
    def __init__(self, tsv_path):
        """
        Load the TSV data into a dict mapping each lowercased word
        to a sorted, deduplicated list of its cleaned IPA entries.
        Cleaning removes ˈ, ː, ., ̯, ̩, and all whitespace.
        """
        df = pd.read_csv(tsv_path, sep='\t', header=0)
        temp = {}
        for _, row in df.iterrows():
            word_val = row['Word']
            ipa_val = row['IPA']
            if pd.isnull(word_val) or pd.isnull(ipa_val):
                continue
            word = str(word_val).lower()
            ipa_raw = str(ipa_val)
            # Remove stress marks, length marks, dots, combining diacritics, and spaces
            ipa_clean = re.sub(r"[ˈː\.\u032F\u0329\s]", "", ipa_raw)
            temp.setdefault(word, set()).add(ipa_clean)
        self.table = {w: sorted(list(ipas)) for w, ipas in temp.items()}

    def get_vocab(self, sentence):
        """
        Extract the lowercased vocabulary from a sentence:
        - Strips out punctuation but retains apostrophes
        - Returns a sorted list of unique words.
        """
        tokens = re.findall(r"[A-Za-z']+", sentence)
        vocab = {token.lower() for token in tokens if re.search(r"[A-Za-z]", token)}
        return sorted(vocab)

    def word_to_ipa_list(self, words):
        """
        Given an iterable of words, returns a list of [word, [possible cleaned IPAs]],
        alphabetized by the word.
        Words not found yield an empty list.
        """
        return [[word, self.table.get(word, [])] for word in sorted(words)]

    def format_string_table(self, words):
        """
        Create a numbered string table in the following format:
        word: IPA
        1. word1: /ipa1/, /ipa2/, ...
        2. word2: /ipa1/, ...
        """
        entries = self.word_to_ipa_list(words)
        lines = []
        for idx, (word, ipas) in enumerate(entries, start=1):
            if len(ipas)==0:
                continue
            ipa_str = ", ".join(f"/{ipa}/" for ipa in ipas) if ipas else ""
            lines.append(f"{idx}. {word}: {ipa_str}")
        return "\n".join(lines)


# Example usage:
if __name__ == "__main__":
    ipa_dict = IpaDictionary('../raw_data/EnglishData.tsv')
    sentence = "Hello, world! It's and zas beautiful day."
    vocab = ipa_dict.get_vocab(sentence)
    table_str = ipa_dict.format_string_table(vocab)
    print(table_str)
