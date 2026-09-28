from __future__ import annotations

import sys
import unicodedata
from functools import lru_cache


def _log(message: str) -> None:
    print(f"[tokenizer-debug] {message}", file=sys.stderr, flush=True)


def apply_tokenizer_patch() -> None:
    """Patch the pure-Python Qwen tokenizer with guarded, observable encoding.

    This keeps the existing tokenizer data/loading code, but replaces the BPE
    loop and encode path so a malformed/non-converging merge cannot hang the
    engine forever. Diagnostics go to stderr, which the Tauri host already
    forwards to the terminal.
    """
    import qwen_engine

    cls = qwen_engine.PurePythonQwen2Tokenizer

    @lru_cache(maxsize=65536)
    def safe_bpe(self, token: str) -> tuple[str, ...]:
        word = tuple(token)
        if len(word) <= 1:
            return word

        # Every valid BPE merge must reduce token count. Therefore len(token)
        # iterations is already a generous upper bound; keep a little slack for
        # diagnostics while guaranteeing termination.
        max_iterations = max(32, len(word) + 8)

        for iteration in range(max_iterations):
            pairs = qwen_engine._get_pairs(word)
            if not pairs:
                return word

            ranked = [
                (self.bpe_ranks[pair], pair)
                for pair in pairs
                if pair in self.bpe_ranks
            ]
            if not ranked:
                return word

            _, (first, second) = min(ranked, key=lambda item: item[0])
            old_len = len(word)
            new_word: list[str] = []
            i = 0

            while i < old_len:
                if i < old_len - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            next_word = tuple(new_word)
            if len(next_word) >= old_len:
                raise RuntimeError(
                    "Qwen BPE merge made no progress: "
                    f"pair=({first!r}, {second!r}), len={old_len}, iteration={iteration}"
                )

            word = next_word
            if len(word) <= 1:
                return word

        raise RuntimeError(
            f"Qwen BPE did not converge after {max_iterations} iterations "
            f"for encoded piece of length {len(token)}"
        )

    def safe_encode_ordinary(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        _log(f"pretokenize:start chars={len(text)}")

        # finditer makes the actual segmentation visible and avoids building
        # nested regex return structures.
        pieces = [match.group(0) for match in self._regex.finditer(text)]
        _log(f"pretokenize:done pieces={len(pieces)}")

        ids: list[int] = []
        for index, piece in enumerate(pieces):
            encoded = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            if index < 5 or index % 25 == 0 or index == len(pieces) - 1:
                _log(
                    f"bpe:piece-start index={index + 1}/{len(pieces)} "
                    f"chars={len(piece)} bytes={len(encoded)}"
                )

            bpe_tokens = self._bpe(encoded)

            for bpe_token in bpe_tokens:
                token_id = self.vocab.get(bpe_token)
                if token_id is None:
                    raise RuntimeError(f"Qwen BPE token missing from vocab: {bpe_token!r}")
                ids.append(int(token_id))

            if index < 5 or index % 25 == 0 or index == len(pieces) - 1:
                _log(
                    f"bpe:piece-done index={index + 1}/{len(pieces)} "
                    f"bpe_tokens={len(bpe_tokens)} total_ids={len(ids)}"
                )

        return ids

    def safe_encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return []

        _log(f"encode:start chars={len(text)} specials={len(self.special_token_ids)}")

        if self.special_pattern is None:
            parts = [text]
        else:
            _log("special-split:start")
            parts = self.special_pattern.split(text)
            _log(f"special-split:done parts={len(parts)}")

        ids: list[int] = []
        ordinary_index = 0
        for part_index, part in enumerate(parts):
            if not part:
                continue

            special_id = self.special_token_ids.get(part)
            if special_id is not None:
                ids.append(special_id)
                _log(
                    f"special-token part={part_index + 1}/{len(parts)} "
                    f"token={part!r} id={special_id}"
                )
                continue

            ordinary_index += 1
            _log(
                f"ordinary-segment:start index={ordinary_index} "
                f"part={part_index + 1}/{len(parts)} chars={len(part)}"
            )
            segment_ids = self._encode_ordinary(part)
            ids.extend(segment_ids)
            _log(
                f"ordinary-segment:done index={ordinary_index} "
                f"ids={len(segment_ids)} total_ids={len(ids)}"
            )

        _log(f"encode:done ids={len(ids)}")
        return ids

    cls._bpe = safe_bpe
    cls._encode_ordinary = safe_encode_ordinary
    cls.encode = safe_encode
    _log("patch-applied")
