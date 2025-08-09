from collections import Counter

import pandas as pd
import json


def load_data_table(filepath):
    """Load TSV file as a DataFrame."""
    return pd.read_csv(filepath, sep='\t')


def generate_sentence_order(df):
    """Generate the 'sentence order' column based on your logic."""

    def process_row(row):
        # If required fields are missing, return empty string
        if pd.isna(row['SentenceLabel']) or pd.isna(row['SentenceSelect']):
            return ''
        try:
            possible_sentences = json.loads(row['possibleSentences'])
            sentence_label = json.loads(row['SentenceLabel'])
            sentence_select = json.loads(row['SentenceSelect'])
        except Exception:
            return ''

        sentences_with_times = []
        label_idx = 0
        for ps in sentence_select:
            value = ps

            if label_idx < len(sentence_label):
                start_time = sentence_label[label_idx]['start']
                label_idx += 1
            else:
                start_time = float('inf')
            sentences_with_times.append((value, start_time))

        sentences_with_times.sort(key=lambda x: x[1])
        ordered_sentences = [s[0] for s in sentences_with_times]

        sentence_order = []
        for sent in ordered_sentences:
            if sent == "Other":
                sentence_order.append(-1)
            else:
                try:
                    idx = possible_sentences.index({'value': sent}) + 1
                    sentence_order.append(idx)
                except ValueError:
                    sentence_order.append(-1)

        return ','.join(str(x) for x in sentence_order)

    # Apply to DataFrame and return
    df['sentence order'] = df.apply(process_row, axis=1)
    return df


def filter_non_monotonic(df):
    """Return rows where sentence order is not monotonically increasing (ignoring -1)."""

    def is_monotonic(row):
        nums = [int(x) for x in row['sentence order'].split(',') if x != '-1' and x != '']
        return all(earlier <= later for earlier, later in zip(nums, nums[1:]))

    # Filter rows where order is NOT monotonic
    return df[~df.apply(is_monotonic, axis=1)]


def find_rows_with_duplicate_non_negative_one_df(df, sentence_order_column="sentence order"):
    """
    Takes a DataFrame and returns rows where a non--1 value
    in the sentence order column is repeated.
    """
    from collections import Counter

    def has_duplicate_non_minus_one(order_str):
        order = str(order_str).split(',')
        nums = [x.strip() for x in order if x.strip() != '-1' and x.strip() != '']
        counts = Counter(nums)
        return any(v > 1 for k, v in counts.items())

    filtered_df = df[df[sentence_order_column].apply(has_duplicate_non_minus_one)]
    return filtered_df


def find_rows_label_longer_than_select(df):
    """
    Returns rows where len(SentenceLabel) > len(SentenceSelect)
    """
    def is_longer(row):
        try:
            labels = json.loads(row['SentenceLabel'])
            selects = json.loads(row['SentenceSelect'])
            return len(labels) > len(selects)
        except Exception:
            return False
    return df[df.apply(is_longer, axis=1)]



if __name__ == "__main__":
    df = load_data_table('../annotationData/sentences/export_157513_project-157513-at-2025-06-29-23-28-82ec7a90.csv')
    df = generate_sentence_order(df)
    filtered_df = filter_non_monotonic(df)
    print(filtered_df)
    df.to_csv('export_157513_project-157513-at-2025-06-29-23-28-82ec7a90_filtered.tsv', index=False,sep='\t')
    df.to_csv('export_157513_project-157513-at-2025-06-29-23-28-82ec7a90_with_order.tsv', index=False,sep='\t')

    duplicates_df=find_rows_with_duplicate_non_negative_one_df(df)
    df.to_csv('export_157513_project-157513-at-2025-06-29-23-28-82ec7a90_with_duplicates.tsv', index=False,sep='\t')

    rows_label_longer = find_rows_label_longer_than_select(df)
    rows_label_longer.to_csv(
        'export_157513_project-157513-at-2025-06-29-23-28-82ec7a90_label_gt_select.tsv',
        sep='\t', index=False
    )
    print(f"Rows where len(SentenceLabel) > len(SentenceSelect): {len(rows_label_longer)}")



