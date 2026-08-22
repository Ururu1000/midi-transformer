"""Bin-packed training dataset with block-diagonal document masking."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class PackedSegment:
    doc_index: int
    row_offset: int
    slot_len: int


class PackedMusicDataset(torch.utils.data.Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Bin-packs whole documents into fixed rows with block-diagonal doc_ids.

    Every document arrives as a set of pre-encoded BPE variants, one per pitch
    shift. Augmentation therefore happens strictly per document, before the
    document lands in a packed row: each segment in a row draws its own shift.
    Row slots are sized from the unshifted variant; a shifted encoding that
    ends up longer is truncated to its slot, a shorter one leaves padding that
    the attention mask and loss both ignore.
    """

    # Sentinel doc id for padding: never equal to any segment ordinal, so real
    # tokens cannot attend into padding.
    PAD_DOC_ID = -1

    def __init__(
        self,
        docs: list[dict[str, Any]],
        pack_seq_len: int,
        pad_token_id: int,
        uncond_id: int,
        cfg_drop_prob: float,
        shifts: list[int],
        augment: bool,
    ) -> None:
        assert docs, "Cannot pack an empty document list"
        assert 0 in shifts, f"Shift variants must include 0, got {shifts}"

        self.docs = docs
        self.pack_seq_len = pack_seq_len
        self.pad_token_id = pad_token_id
        self.uncond_id = uncond_id
        self.cfg_drop_prob = cfg_drop_prob
        self.shifts = list(shifts)
        self.augment = augment

        self.rows: list[list[PackedSegment]] = []
        current_row: list[PackedSegment] = []
        cursor = 0
        for doc_index, doc in enumerate(self.docs):
            # +1 for the composer prefix written at __getitem__ time.
            slot_len = min(int(doc["variants"][0].numel()) + 1, pack_seq_len)
            if cursor + slot_len > pack_seq_len:
                self.rows.append(current_row)
                current_row = []
                cursor = 0
            current_row.append(PackedSegment(doc_index, cursor, slot_len))
            cursor += slot_len
        if current_row:
            self.rows.append(current_row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        row = torch.full((self.pack_seq_len,), self.pad_token_id, dtype=torch.long)
        doc_ids = torch.full((self.pack_seq_len,), self.PAD_DOC_ID, dtype=torch.long)
        # One coin per row: the whole row is either conditioned or unconditioned,
        # matching how classifier-free guidance runs at inference.
        drop_conditioning = (
            self.cfg_drop_prob > 0.0 and random.random() < self.cfg_drop_prob
        )

        segments = self.rows[index]
        for ordinal, segment in enumerate(segments):
            doc = self.docs[segment.doc_index]
            shift = random.choice(self.shifts) if self.augment else 0
            prefix_id = self.uncond_id if drop_conditioning else int(doc["composer_id"])
            variant = doc["variants"][shift].long()
            content = torch.cat(
                [torch.tensor([prefix_id], dtype=torch.long), variant]
            )
            fit = min(int(content.numel()), segment.slot_len)
            start = segment.row_offset
            row[start : start + fit] = content[:fit]
            doc_ids[start : start + fit] = ordinal

        inputs = row[:-1]
        targets = row[1:].clone()
        input_doc_ids = doc_ids[:-1]
        # The position just before a document start would otherwise be scored on
        # predicting the next document's composer prefix.
        for segment in segments:
            if segment.row_offset > 0:
                targets[segment.row_offset - 1] = self.pad_token_id

        expected_shape = (self.pack_seq_len - 1,)
        assert inputs.shape == expected_shape, f"Got {inputs.shape}"
        assert targets.shape == expected_shape, f"Got {targets.shape}"
        assert input_doc_ids.shape == expected_shape, f"Got {input_doc_ids.shape}"
        return inputs, targets, input_doc_ids
