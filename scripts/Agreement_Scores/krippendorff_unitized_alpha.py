"""
Python module to compute krippendorff's unitized alpha

https://link.springer.com/article/10.1007/s11135-015-0266-1

source: https://github.com/zeljkobekcic/KrippendorffUnitizedAlpha/tree/main
"""

import numpy as np
from dataclasses import dataclass
from itertools import product, tee


EMPTY_TAG = "_empty"


@dataclass(frozen=True)
class Segment:
    tag: str
    start: int
    end: int

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError(
                f"start {self.start} is not allowed to be equal or larger than end {self.end}"
            )

    @property
    def length(self) -> int:
        return self.end - self.start

    def intersect_length(self, other):
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        length = max(end - start, 0)
        return length


class KrippendorffUnitizedAlpha:
    def __init__(self, data: dict[str, list[Segment]]):
        self.data = data
        self.m = len(data)
        self.tag_set = self._get_all_tags()
        self.tag_2_pos: dict[str, int] = {tag: i for i, tag in enumerate(self.tag_set)}
        self.c_k_combinations = list(product(self.tag_set, self.tag_set))

        self.observed_coincidence_matrix_ = self.calculate_observed_coincidences_matrix()
        self.l_dot_dot_ = self.observed_coincidence_matrix_.sum()
        self.l_dot_tag_ = self.observed_coincidence_matrix_.sum(axis=0)
        self.l_tag_dot_ = self.observed_coincidence_matrix_.sum(axis=1)

        self.expected_coincidence_matrix_ = self.calculate_expected_coincidence_matrix()
        self.result_ = self.calculate_krippendorff_unitized_alpha()

    def _get_all_tags(self) -> set[str]:
        return {segment.tag for annotations in self.data.values() for segment in annotations}

    def calculate_observed_coincidence_matrix_field(self, tag_c: str, tag_k: str) -> float:
        acc = []
        _data_c = {coder: [s for s in annotation if s.tag == tag_c] for coder, annotation in
                   self.data.items()}
        _data_k = {coder: [s for s in annotation if s.tag == tag_k] for coder, annotation in
                   self.data.items()}

        for c1, a1 in _data_c.items():
            for c2, a2 in _data_k.items():
                if c1 != c2:
                    for s1 in a1:
                        for s2 in a2:
                            x = s1.intersect_length(s2)
                            acc.append(x)

        return (1 / (self.m - 1)) * sum(acc)

    def calculate_observed_coincidences_matrix(self) -> np.ndarray:
        num_tags = len(self.tag_set)
        observed_coincidences = np.zeros((num_tags, num_tags))

        for c, k in self.c_k_combinations:
            i, j = self.tag_2_pos[c], self.tag_2_pos[k]
            observed_coincidences[i, j] = self.calculate_observed_coincidence_matrix_field(c, k)

        return observed_coincidences

    def calculate_expected_coincidence_matrix(self) -> np.ndarray:
        num_tags = len(self.tag_set)
        expected_coincidences = np.zeros((num_tags, num_tags))

        for c, k in self.c_k_combinations:
            i, j = self.tag_2_pos[c], self.tag_2_pos[k]

            acc_denominator = []
            for coder, annotation in self.data.items():
                for s in annotation:
                    acc_denominator.append(s.length if s.tag == EMPTY_TAG else s.length ** 2)

            denominator = self.l_dot_dot_ ** 2 - sum(acc_denominator)

            acc_enumerator = []

            # sum over segment length
            for coder, annotation in self.data.items():
                for s in annotation:
                    if c == k == s.tag:
                        if s.tag == EMPTY_TAG:
                            acc_enumerator.append(s.length)
                        else:
                            acc_enumerator.append(s.length ** 2)
            enumerator = self.l_tag_dot_[i] * self.l_dot_tag_[j] - sum(acc_enumerator)

            expected_coincidences[i, j] = self.l_dot_dot_ * enumerator / denominator

        return expected_coincidences

    def calculate_krippendorff_unitized_alpha(self):
        observed = self.l_dot_dot_ - (self.observed_coincidence_matrix_.diagonal().sum())
        expected = self.l_dot_dot_ - (self.expected_coincidence_matrix_.diagonal().sum())
        return 1 - (observed / expected)

    @property
    def result(self) -> float:
        return self.result_


def impute_empty_segments(data: list[Segment], max_length: int) -> list[Segment]:
    sorted_data = sorted(data, key=lambda x: x.start)
    imputed_data = []

    if not sorted_data:
        return [Segment(tag=EMPTY_TAG, start=0, end=max_length-1)]

    current_segments, next_segments = tee(sorted_data)

    first_element = next(next_segments)
    if first_element.start != 0 and 0 < first_element.start - 1:
        imputed_data.append(Segment(tag=EMPTY_TAG, start=0, end=first_element.start - 1))

    for cur_segment, next_segment in zip(current_segments, next_segments):

        if (cur_segment.end+1) == next_segment.start:
            imputed_data.append(cur_segment)
        else:
            imputed_data.append(cur_segment)
            imputed_data.append(
                Segment(tag=EMPTY_TAG, start=cur_segment.end+1 ,end=next_segment.start-1)
            )

    if len(sorted_data) == 1:
        last_element = first_element
        imputed_data.append(last_element)
    else:
        last_element = next_segment
        imputed_data.append(last_element)

    if last_element.end + 1 < max_length and last_element.end + 1< max_length - 1:
        imputed_data.append(Segment(tag=EMPTY_TAG, start=last_element.end + 1, end=max_length - 1))

    return imputed_data
