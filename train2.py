# Import necessary modules and libraries
from model import build_transformer  # Import the build_transformer function from the model module
from dataset import BilingualDataset, causal_mask  # Import BilingualDataset and causal_mask from the dataset module
from config import get_config, get_weights_file_path, latest_weights_file_path  # Import configuration and file path functions from the config module

import torchtext.datasets as datasets  # Import torchtext datasets
import torch  # Import PyTorch
import torch.nn as nn  # Import neural network modules from PyTorch
from torch.utils.data import Dataset, DataLoader, random_split  # Import Dataset, DataLoader, and random_split from PyTorch
from torch.optim.lr_scheduler import LambdaLR  # Import LambdaLR scheduler from PyTorch

import warnings  # Import warnings to manage warning messages
from tqdm import tqdm  # Import tqdm for progress bars
import os  # Import os for operating system interactions
from pathlib import Path  # Import Path from pathlib for path manipulations

# Huggingface datasets and tokenizers
from datasets import load_dataset  # Import load_dataset from Huggingface datasets
from tokenizers import Tokenizer  # Import Tokenizer from Huggingface tokenizers
from tokenizers.models import WordLevel  # Import WordLevel model from Huggingface tokenizers
from tokenizers.trainers import WordLevelTrainer  # Import WordLevelTrainer from Huggingface tokenizers
from tokenizers.pre_tokenizers import Whitespace  # Import Whitespace pre-tokenizer from Huggingface tokenizers

import torchmetrics  # Import torchmetrics for evaluation metrics
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter for TensorBoard logging

# Function to perform greedy decoding using the transformer model-used for validation purpose only

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Perform greedy decoding to generate a translation given a source sentence.
    
    Args:
    - model (nn.Module): Transformer model for sequence-to-sequence tasks.
    - source (torch.Tensor): Input tensor representing the source sentence.
    - source_mask (torch.Tensor): Mask tensor for the source input to handle padding.
    - tokenizer_src (Tokenizer): Tokenizer for source language.
    - tokenizer_tgt (Tokenizer): Tokenizer for target language.
    - max_len (int): Maximum length for the generated target sequence.
    - device (torch.device): Device on which the computation should be performed.
    
    Returns:
    - torch.Tensor: Tensor representing the decoded target sentence.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')  # Start of sentence token index
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')  # End of sentence token index

    # Pass the source sentence tensor through encode in model.py
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    """
    Run validation on the model and calculate evaluation metrics.
    
    Args:
    - model (nn.Module): Transformer model for sequence-to-sequence tasks.
    - validation_ds (DataLoader): Validation dataset loader.
    - tokenizer_src (Tokenizer): Tokenizer for source language.
    - tokenizer_tgt (Tokenizer): Tokenizer for target language.
    - max_len (int): Maximum length for the generated target sequence.
    - device (torch.device): Device on which the computation should be performed.
    - print_msg (function): Function to print messages during validation.
    - global_step (int): Current global step in training.
    - writer (SummaryWriter): TensorBoard writer for logging.
    - num_examples (int): Number of examples to validate.
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # Get the console window width for formatting output
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # Default console width if retrieval fails
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # Check that the batch size is 1 for validation
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Perform greedy decoding to get the model's output
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Retrieve source, target, and predicted texts
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target, and predicted texts
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    '''
    # Evaluation metrics (commented out for now)
    if writer:
        # Evaluate the character error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Evaluate the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Evaluate the BLEU score
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
    '''


def get_all_sentences(ds, lang):
    """
    Generator function to yield all sentences from a dataset for a given language.
    
    Args:
    - ds (Dataset): Dataset containing translations.
    - lang (str): Language code for the translations.
    """
    for item in ds:
        yield item['translation'][lang]

#Forming the tokenizers for english(input) & italian language(output)
def get_or_build_tokenizer(config, ds, lang):
    """
    Retrieve or build a tokenizer for a specified language.
    
    Args:
    - config (dict): Configuration dictionary containing tokenizer file paths.
    - ds (Dataset): Dataset containing translations.
    - lang (str): Language code for the translations. "en" for english and "it" for italian
    
    Returns:
    - Tokenizer: Tokenizer object for the specified language.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) #note the tokenizer json files formed to store tokenizer for english & italian
    if not Path.exists(tokenizer_path):
        # Initialize a WordLevel tokenizer with an unknown token "[UNK]"
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        # Set the pre-tokenizer to split tokens based on whitespace
        tokenizer.pre_tokenizer = Whitespace()
        # Define the WordLevelTrainer with special tokens and minimum frequency
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # Train the tokenizer from the iterator of sentences in the dataset
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        # Save the trained tokenizer to the specified file path
        tokenizer.save(str(tokenizer_path))
    else:
        # Tokenizer already exists, so load it from file
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """
    Prepare train and validation datasets along with tokenizers.
    
    Args:
    - config (dict): Configuration dictionary containing dataset and training parameters.
    
    Returns:
    - DataLoader: DataLoader for training dataset.
    - DataLoader: DataLoader for validation dataset.
    - Tokenizer: Tokenizer for source language.
    - Tokenizer: Tokenizer for target language.
    """
    # Load dataset (only train split available)
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers for source and target languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split dataset into 90% training and 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Create BilingualDataset instances for training and validation
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find maximum lengths of source and target sentences in the dataset
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Create DataLoader instances for training and validation datasets
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)  #for validation data keep batch size 1

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Initialize the Transformer model for sequence-to-sequence tasks.
    
    Args:
    - config (dict): Configuration dictionary containing model parameters.
    - vocab_src_len (int): Size of the vocabulary for source language.
    - vocab_tgt_len (int): Size of the vocabulary for target language.
    
    Returns:
    - nn.Module: Initialized Transformer model.
    """
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model


def train_model(config):
    """
    Train the Transformer model for sequence-to-sequence tasks.
    
    Args:
    - config (dict): Configuration dictionary containing training parameters.
    """
    # Determine device for computation (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)

    # Ensure weights folder exists for model checkpoints
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Prepare train and validation datasets, along with tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Initialize Transformer model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Initialize Tensorboard for visualization
    writer = SummaryWriter(config['experiment_name'])

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Load pre-trained model if specified
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Training loop over epochs
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        # Wrap the DataLoader with tqdm to create a progress bar
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            #encoder_input, decoder_input,encoder_mask, decoder_mask were all returned by BilingualDataset class  __getitem__ function.
            #DataLoader gives us these exact corresponding to the current batch of train_data we are using
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # Forward pass through encoder, decoder, and projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Calculate the loss using cross entropy
            label = batch['label'].to(device)  # (B, seq_len)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) #calc loss of proj_output wrt label
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the training loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of each epoch (uncomment when needed)
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save model checkpoint at the end of each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    # Ignore warnings during execution
    warnings.filterwarnings("ignore")

    # Load configuration settings
    config = get_config()

    # Start training the Transformer model
    train_model(config)