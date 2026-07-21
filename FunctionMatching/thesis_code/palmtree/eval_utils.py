import torch
import re

import palmtree.vocab as vocab


PALMTREE_POOLING_MEAN = "mean"
PALMTREE_POOLING_MASKED_MEAN = "masked_mean"
PALMTREE_POOLING_MODES = (
    PALMTREE_POOLING_MEAN,
    PALMTREE_POOLING_MASKED_MEAN,
)
PALMTREE_SEQUENCE_LENGTH = 20


# this function is how I parse and pre-pocess instructions for palmtree. It is very simple and based on regular expressions. 
# If I use IDA pro or angr instead of Binary Ninja, I would have come up with a better solution.

def parse_instruction(ins, symbol_map, string_map):
    # arguments:
    # ins: string e.g. "mov, eax, [rax+0x1]"
    # symbol_map: a dict that contains symbols the key is the address and the value is the symbol 
    # string_map : same as symbol_map in Binary Ninja, constant strings will be included into string_map 
    #              and the other meaningful strings like function names will be included into the symbol_map
    #              I think you do not have to separate them. This is just one of the possible nomailization stretagies.
    ins = re.sub(r'\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    token_lst = []
    if len(parts) > 1:
        operand = parts[1:]
    token_lst.append(parts[0])
    for i in range(len(operand)):
        # print(operand)
        symbols = re.split(r'([0-9A-Za-z]+)', operand[i])
        symbols = [s.strip() for s in symbols if s]
        processed = []
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) > 6 and len(symbols[j]) < 15: 
                # I make a very dumb rule here to treat number larger than 6 but smaller than 15 digits as addresses, 
                # the others are constant numbers and will not be normalized.
                if int(symbols[j], 16) in symbol_map:
                    processed.append("symbol")
                elif int(symbols[j], 16) in string_map:
                    processed.append("string")
                else:
                    processed.append("address")
            else:
                processed.append(symbols[j])
            processed = [p for p in processed if p]

        token_lst.extend(processed) 

    # the output will be like "mov eax [ rax + 0x1 ]"
    return ' '.join(token_lst)



class UsableTransformer:
    def __init__(
        self,
        model_path,
        vocab_path,
        device="auto",
        pooling=PALMTREE_POOLING_MASKED_MEAN,
    ):
        print("Loading Vocab", vocab_path)
        self.vocab = vocab.WordVocab.load_vocab(vocab_path)
        print("Vocab Size: ", len(self.vocab))
        self.device = self._resolve_device(device)
        if pooling not in PALMTREE_POOLING_MODES:
            raise ValueError(
                f"Unknown PalmTree pooling '{pooling}'; "
                f"expected one of {PALMTREE_POOLING_MODES}"
            )
        self.pooling = pooling
        self.model = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device):
        requested = str(device or "auto").strip().lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        resolved = torch.device(requested)
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"PyTorch device '{requested}' requested, but CUDA is unavailable"
                )
            if resolved.index is not None and resolved.index >= torch.cuda.device_count():
                raise ValueError(
                    f"PyTorch device '{requested}' does not exist; "
                    f"found {torch.cuda.device_count()} CUDA device(s)"
                )
        return resolved

    def _instruction_tensors(self, text):
        """Build PalmTree inputs while preserving the selected adapter semantics."""
        segment_labels = []
        sequences = []
        for instruction in text:
            token_ids = self.vocab.to_seq(instruction)
            if self.pooling == PALMTREE_POOLING_MEAN:
                # Reproduce the original adapter exactly for old experiments.
                sequence = [self.vocab.sos_index] + token_ids + [self.vocab.eos_index]
                valid_length = len(instruction.split(" ")) + 2
                sequence = sequence[:PALMTREE_SEQUENCE_LENGTH]
                segment_label = [1] * min(
                    valid_length,
                    PALMTREE_SEQUENCE_LENGTH,
                )
            else:
                # Reserve one position for both boundary tokens.  Truncating the
                # content first prevents long instructions from losing <eos>.
                content_length = PALMTREE_SEQUENCE_LENGTH - 2
                sequence = (
                    [self.vocab.sos_index]
                    + token_ids[:content_length]
                    + [self.vocab.eos_index]
                )
                segment_label = [1] * len(sequence)

            sequence_padding = PALMTREE_SEQUENCE_LENGTH - len(sequence)
            segment_padding = PALMTREE_SEQUENCE_LENGTH - len(segment_label)
            sequences.append(
                sequence + [self.vocab.pad_index] * sequence_padding
            )
            segment_labels.append(segment_label + [0] * segment_padding)

        segment_label = torch.tensor(
            segment_labels,
            dtype=torch.long,
            device=self.device,
        )
        sequence = torch.tensor(
            sequences,
            dtype=torch.long,
            device=self.device,
        )
        return sequence, segment_label

    def _pool_encoded(self, encoded, sequence):
        if self.pooling == PALMTREE_POOLING_MEAN:
            return torch.mean(encoded, dim=1)

        valid = sequence.ne(self.vocab.pad_index).unsqueeze(-1)
        weights = valid.to(dtype=encoded.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (encoded * weights).sum(dim=1) / denominator


    def encode(self, text, output_option='lst'):
        sequence, segment_label = self._instruction_tensors(text)

        with torch.inference_mode():
            encoded = self.model.forward(sequence, segment_label)
            result = self._pool_encoded(encoded, sequence)

        return result.detach().cpu().numpy()
