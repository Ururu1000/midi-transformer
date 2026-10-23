import pytest

from musiclm.data.preprocess import chunk_base_ids


class TestChunkBaseIds:
    def test_chunks_leave_room_for_composer_prefix(self):
        chunks = chunk_base_ids(list(range(20)), max_seq_len=8, stride=7)
        assert [len(chunk) for chunk in chunks] == [7, 7, 6]
        assert [token for chunk in chunks for token in chunk] == list(range(20))

    def test_overlapping_stride(self):
        chunks = chunk_base_ids(list(range(10)), max_seq_len=6, stride=3)
        assert chunks[0] == [0, 1, 2, 3, 4]
        assert chunks[1][:2] == [3, 4]

    def test_short_sequence_is_one_chunk(self):
        assert chunk_base_ids([5, 6], max_seq_len=8, stride=7) == [[5, 6]]

    def test_stride_beyond_content_rejected(self):
        with pytest.raises(AssertionError):
            chunk_base_ids(list(range(10)), max_seq_len=8, stride=8)
