import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Custom dataset class for bilingual translation
class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len  # Maximum sequence length for input and output

        self.ds = ds  # The dataset containing translation pairs
        self.tokenizer_src = tokenizer_src  # Tokenizer for source language
        self.tokenizer_tgt = tokenizer_tgt  # Tokenizer for target language
        self.src_lang = src_lang  # Source language key
        self.tgt_lang = tgt_lang  # Target language key

        # Special tokens
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)  # Start of sentence token
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)  # End of sentence token
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)  # Padding token

    def __len__(self):
        return len(self.ds)  # Return the length of the dataset

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]  # Get the source-target pair from the dataset
        src_text = src_target_pair['translation'][self.src_lang]  # Extract source text
        tgt_text = src_target_pair['translation'][self.tgt_lang]  # Extract target text

        # Encode the text into token ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate the number of padding tokens required-required to decide how many padding tokens to append before encoding 
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # For <s> and </s>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # For <s>

        # Ensure the sentence is not too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Create encoder input by adding <s>, </s>, and padding tokens
        encoder_input = torch.cat(
            [
                self.sos_token,  # Start-of-sequence token (tensor)
                torch.tensor(enc_input_tokens, dtype=torch.int64),  # Tokenized input sequence (tensor)
                self.eos_token,  # End-of-sequence token (tensor)
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),  # Padding tokens to reach fixed sequence length (tensor)
            ],
            dim=0,  # Concatenate along the first dimension (sequence length)
        )

        # Create decoder input by adding <s> and padding tokens
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create label by adding </s> and padding tokens
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Ensure the tensors are of the correct length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Return the prepared tokenized encoder and decoder inputs and original src and tgt texts
        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

# Function to create a causal mask for the decoder
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)  # Upper triangular matrix with 1s above the diagonal
    return mask == 0  # Convert to boolean mask where 0 indicates allowed positions
