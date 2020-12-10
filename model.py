"""Model classes and model utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""
import string
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD_TOKEN, UNK_TOKEN
from utils import cuda, load_cached_embeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _sort_batch_by_length(tensor, sequence_lengths):
    """
    Sorts input sequences by lengths. This is required by Pytorch
    `pack_padded_sequence`. Note: `pack_padded_sequence` has an option to
    sort sequences internally, but we do it by ourselves.

    Args:
        tensor: Input tensor to RNN [batch_size, len, dim].
        sequence_lengths: Lengths of input sequences.

    Returns:
        sorted_tensor: Sorted input tensor ready for RNN [batch_size, len, dim].
        sorted_sequence_lengths: Sorted lengths.
        restoration_indices: Indices to recover the original order.
    """
    # Sort sequence lengths
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    # Sort sequences
    sorted_tensor = tensor.index_select(0, permutation_index)
    # Find indices to recover the original order
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths))).long()
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices


class AlignedAttention(nn.Module):
    """
    This module returns attention scores over question sequences. Details can be
    found in these papers:
        - Aligned question embedding (Chen et al. 2017):
             https://arxiv.org/pdf/1704.00051.pdf
        - Context2Query (Seo et al. 2017):
             https://arxiv.org/pdf/1611.01603.pdf

    Args:
        p_dim: Int. Passage vector dimension.

    Inputs:
        p: Passage tensor (float), [batch_size, p_len, p_dim].
        q: Question tensor (float), [batch_size, q_len, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Attention scores over question sequences, [batch_size, p_len, q_len].
    """

    def __init__(self, p_dim):
        super().__init__()
        self.linear = nn.Linear(p_dim, p_dim)
        self.relu = nn.ReLU()

    def forward(self, p, q, q_mask):
        # Compute scores
        p_key = self.relu(self.linear(p))  # [batch_size, p_len, p_dim]
        q_key = self.relu(self.linear(q))  # [batch_size, q_len, p_dim]
        scores = p_key.bmm(q_key.transpose(2, 1))  # [batch_size, p_len, q_len]
        # Stack question mask p_len times
        q_mask = q_mask.unsqueeze(1).repeat(1, scores.size(1), 1)
        # Assign -inf to pad tokens
        scores.data.masked_fill_(q_mask.data, -float('inf'))
        # Normalize along question length
        return F.softmax(scores, 2)  # [batch_size, p_len, q_len]


class SpanAttention(nn.Module):
    """
    This module returns attention scores over sequence length.

    Args:
        q_dim: Int. Passage vector dimension.

    Inputs:
        q: Question tensor (float), [batch_size, q_len, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Attention scores over sequence length, [batch_size, len].
    """

    def __init__(self, q_dim):
        super().__init__()
        self.linear = nn.Linear(q_dim, 1)

    def forward(self, q, q_mask):
        # Compute scores
        q_scores = self.linear(q).squeeze(2)  # [batch_size, len]
        # Assign -inf to pad tokens
        q_scores.data.masked_fill_(q_mask.data, -float('inf'))
        # Normalize along sequence length
        return F.softmax(q_scores, 1)  # [batch_size, len]


class BilinearOutput(nn.Module):
    """
    This module returns logits over the input sequence.

    Args:
        p_dim: Int. Passage hidden dimension.
        q_dim: Int. Question hidden dimension.

    Inputs:
        p: Passage hidden tensor (float), [batch_size, p_len, p_dim].
        q: Question vector tensor (float), [batch_size, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Logits over the input sequence, [batch_size, p_len].
    """

    def __init__(self, p_dim, q_dim):
        super().__init__()
        self.linear = nn.Linear(q_dim, p_dim)

    def forward(self, p, q, p_mask):
        # Compute bilinear scores
        q_key = self.linear(q).unsqueeze(2)  # [batch_size, p_dim, 1]
        p_scores = torch.bmm(p, q_key).squeeze(2)  # [batch_size, p_len]
        # Assign -inf to pad tokens
        p_scores.data.masked_fill_(p_mask.data, -float('inf'))
        return p_scores  # [batch_size, p_len]


class LetterEmbeddings(nn.Module):
    # [ for padding and ] for unknown
    pad_token = '['
    unk_token = '['
    vocab = pad_token + unk_token + string.ascii_lowercase + ' ?!-_'
    idx_pad = 0
    idx_unk = 1
    max_length = 30

    def __init__(self, embedding_dim=25, use_gpu=True):
        super().__init__()
        # Index 0 is for padding and 1 is for an unknown char
        # Padding just returns an embedding of 0
        self.net = nn.Embedding(len(self.vocab) + 1, embedding_dim, scale_grad_by_freq=True)
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if use_gpu else 'cpu')

    def forward(self, letters: torch.LongTensor):
        z = torch.zeros(*letters.shape, self.embedding_dim, device=self.device)
        z[letters != 0, :] = self.net(letters[letters != 0])
        return self.net(letters)

    def get_embeddings(self, word_embeddings: torch.LongTensor, sentences: List[List[str]]):
        shape = word_embeddings.shape
        words = ''
        lengths = torch.ones(shape[0] * shape[1], device=self.device)
        for i in range(shape[0]):
            n = 0
            for j in range(shape[1]):
                embed = word_embeddings[i, j]
                if embed == 0:
                    words += (self.max_length * self.pad_token)
                elif embed == 1:
                    words += (self.max_length * self.unk_token)
                else:
                    word = sentences[i][n]
                    words += (word + (self.max_length - len(word)) * self.pad_token)
                    lengths[i * j] = len(word)
                n += 1

        x = torch.ones(len(words), dtype=torch.long, device=self.device)
        idx = torch.as_tensor(np.array(list(words))[:, None] == np.array(list(self.vocab))[None], device=self.device)\
            .nonzero()
        x[idx[:, 0]] = idx[:, 1]

        # return self(x.reshape(*shape, self.max_length)).mean(2)
        # Instead of mean, take sum and divide by lengths to not penalize small words
        return self(x.reshape(*shape, self.max_length)).sum(2) / lengths.reshape(*shape)[..., None]


class BaselineReader(nn.Module):
    """
    Baseline QA Model
    [Architecture]
        0) Inputs: passages and questions
        1) Embedding Layer: converts words to vectors
        2) Context2Query: computes weighted sum of question embeddings for
               each position in passage.
        3) Passage Encoder: LSTM or GRU.
        4) Question Encoder: LSTM or GRU.
        5) Question Attentive Sum: computes weighted sum of question hidden.
        6) Start Position Pointer: computes scores (logits) over passage
               conditioned on the question vector.
        7) End Position Pointer: computes scores (logits) over passage
               conditioned on the question vector.

    Args:
        args: `argparse` object.

    Inputs:
        batch: a dictionary containing batched tensors.
            {
                'passages': LongTensor [batch_size, p_len],
                'questions': LongTensor [batch_size, q_len],
                'start_positions': Not used in `forward`,
                'end_positions': Not used in `forward`,
            }

    Returns:
        Logits for start positions and logits for end positions.
        Tuple: ([batch_size, p_len], [batch_size, p_len])
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.pad_token_id = args.pad_token_id

        # Initialize embedding layer (1)
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)  # word embeddings
        self.letter_embedding = None    # letter embeddings
        if args.letter_embedding_dim > 0:
            self.letter_embedding = LetterEmbeddings(args.letter_embedding_dim, args.use_gpu)

        # Initialize Context2Query (2)
        self.aligned_att = AlignedAttention(args.embedding_dim + args.letter_embedding_dim)

        rnn_cell = nn.LSTM if args.rnn_cell_type == 'lstm' else nn.GRU

        # Initialize passage encoder (3)
        self.passage_rnn = rnn_cell(
            (args.embedding_dim + args.letter_embedding_dim) * 2,
            args.hidden_dim,
            bidirectional=args.bidirectional,
            batch_first=True,
        )

        # Initialize question encoder (4)
        self.question_rnn = rnn_cell(
            args.embedding_dim + args.letter_embedding_dim,
            args.hidden_dim,
            bidirectional=args.bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(self.args.dropout)

        # Adjust hidden dimension if bidirectional RNNs are used
        _hidden_dim = (
            args.hidden_dim * 2 if args.bidirectional
            else args.hidden_dim
        )

        # Initialize attention layer for question attentive sum (5)
        self.question_att = SpanAttention(_hidden_dim)

        # Initialize bilinear layer for start positions (6)
        self.start_output = BilinearOutput(_hidden_dim, _hidden_dim)

        # Initialize bilinear layer for end positions (7)
        self.end_output = BilinearOutput(_hidden_dim, _hidden_dim)

    def load_pretrained_embeddings(self, vocabulary, path):
        """
        Loads GloVe vectors and initializes the embedding matrix.

        Args:
            vocabulary: `Vocabulary` object.
            path: Embedding path, e.g. "glove/glove.6B.300d.txt".
        """
        embedding_map = load_cached_embeddings(path)

        # Create embedding matrix. By default, embeddings are randomly
        # initialized from Uniform(-0.1, 0.1).
        embeddings = torch.zeros(
            (len(vocabulary), self.args.embedding_dim)
        ).uniform_(-0.1, 0.1)

        # Initialize pre-trained embeddings.
        num_pretrained = 0
        for (i, word) in enumerate(vocabulary.words):
            if word in embedding_map:
                embeddings[i] = torch.tensor(embedding_map[word])
                num_pretrained += 1

        # Place embedding matrix on GPU.
        self.embedding.weight.data = cuda(self.args, embeddings)

        return num_pretrained

    def sorted_rnn(self, sequences, sequence_lengths, rnn):
        """
        Sorts and packs inputs, then feeds them into RNN.

        Args:
            sequences: Input sequences, [batch_size, len, dim].
            sequence_lengths: Lengths for each sequence, [batch_size].
            rnn: Registered LSTM or GRU.

        Returns:
            All hidden states, [batch_size, len, hid].
        """
        # Sort input sequences
        sorted_inputs, sorted_sequence_lengths, restoration_indices = _sort_batch_by_length(
            sequences, sequence_lengths
        )
        # Pack input sequences
        packed_sequence_input = pack_padded_sequence(
            sorted_inputs,
            sorted_sequence_lengths.data.long().tolist(),
            batch_first=True
        )
        # Run RNN
        packed_sequence_output, _ = rnn(packed_sequence_input, None)
        # Unpack hidden states
        unpacked_sequence_tensor, _ = pad_packed_sequence(
            packed_sequence_output, batch_first=True
        )
        # Restore the original order in the batch and return all hidden states
        return unpacked_sequence_tensor.index_select(0, restoration_indices)

    def forward(self, batch):
        # Obtain masks and lengths for passage and question.
        passage_mask = (batch['passages'] != self.pad_token_id)  # [batch_size, p_len]
        question_mask = (batch['questions'] != self.pad_token_id)  # [batch_size, q_len]
        passage_lengths = passage_mask.long().sum(-1)  # [batch_size]
        question_lengths = question_mask.long().sum(-1)  # [batch_size]

        # 1) Embedding Layer: Embed the passage and question.
        passage_embeddings = self.embedding(batch['passages'])  # [batch_size, p_len, p_dim]
        question_embeddings = self.embedding(batch['questions'])  # [batch_size, q_len, q_dim]
        if self.letter_embedding is not None:
            # [batch_size, p_len, p_dim + letter_embeddings]
            passage_embeddings = torch.cat((passage_embeddings,
                                            self.letter_embedding.get_embeddings(batch['passages'],
                                                                                 batch['passage_words'])), 2)
            # [batch_size, q_len, q_dim + letter_embeddings]
            question_embeddings = torch.cat((question_embeddings,
                                             self.letter_embedding.get_embeddings(batch['questions'],
                                                                                  batch['question_words'])), 2)

        # 2) Context2Query: Compute weighted sum of question embeddings for
        #        each passage word and concatenate with passage embeddings.
        aligned_scores = self.aligned_att(
            passage_embeddings, question_embeddings, ~question_mask
        )  # [batch_size, p_len, q_len]
        aligned_embeddings = aligned_scores.bmm(question_embeddings)  # [batch_size, p_len, q_dim + letter_embeddings]
        passage_embeddings = cuda(
            self.args,
            torch.cat((passage_embeddings, aligned_embeddings), 2),
        )  # [batch_size, p_len, p_dim + q_dim + 2 * letter_embeddings]

        # 3) Passage Encoder
        passage_hidden = self.sorted_rnn(
            passage_embeddings, passage_lengths, self.passage_rnn
        )  # [batch_size, p_len, p_hid]
        passage_hidden = self.dropout(passage_hidden)  # [batch_size, p_len, p_hid]

        # 4) Question Encoder: Encode question embeddings.
        question_hidden = self.sorted_rnn(
            question_embeddings, question_lengths, self.question_rnn
        )  # [batch_size, q_len, q_hid]

        # 5) Question Attentive Sum: Compute weighted sum of question hidden
        #        vectors.
        question_scores = self.question_att(question_hidden, ~question_mask)
        question_vector = question_scores.unsqueeze(1).bmm(question_hidden).squeeze(1)
        question_vector = self.dropout(question_vector)  # [batch_size, q_hid]

        # 6) Start Position Pointer: Compute logits for start positions
        start_logits = self.start_output(
            passage_hidden, question_vector, ~passage_mask
        )  # [batch_size, p_len]

        # 7) End Position Pointer: Compute logits for end positions
        end_logits = self.end_output(
            passage_hidden, question_vector, ~passage_mask
        )  # [batch_size, p_len]

        return start_logits, end_logits  # [batch_size, p_len], [batch_size, p_len]
