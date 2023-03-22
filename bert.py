from typing import Dict, List, Optional, Union, Tuple, BinaryIO
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from dfp_utils import *
import numpy as np
#import transformers
#import transformers.Constants as Constants
#from transformers.Layers import DecoderLayer


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this dropout is applied to normalized attention scores following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj


  #the following helper function is used for whenever we need to normalize a vector
  #https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
  def normalize(v):
     norm = np.linalg.norm(v)
     if norm == 0:
       return v
     return v / norm

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 

    # normalize the scores
    # multiply the attention scores to the value and get back V'
    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]


    ##go back and add comments on the shapes Daniel!!!!!! 
    #attention score calculation

    # print("Query size: " + str(query.shape)) # ->   Query size: torch.Size([2, 12, 8, 64])
    # print("Key size: " + str(key.shape)) # ->   Key size: torch.Size([2, 12, 8, 64])

    #SHAPES: (???) @ (???).T -> (bs, num_attention_heads, seq_len, seq_len)
    # I think we don't need to transpose since they've already been transformed linearly?
    #S = torch.matmul(query, torch.reshape(key, (2, 12, 64, 8)))
    #print("S size: " + str(S.shape)) # ->   S size: torch.Size([2, 12, 8, 8])
    # print("Attention mask size: " + str(attention_mask.shape)) # ->   Attention mask size: torch.Size([2, 1, 1, 8])

    # Apply mask
    # or S.permute(*torch.arrange(x.ndim - 1, -1, -1))
    #S = S @ torch.reshape(attention_mask, (1, 1, 8, 2))
    # Normalize the scores
    # sqrt(d_k)
    #norm_S = S * (1.0 / math.sqrt(key.size(-1)))
    # S has shape: [2, 12, 8, 8]
    # Multiply attention scores to value

    # value has shape: [2, 12, 8, 64]
    #softmax = nn.Softmax(dim=3)
    #value_prime = torch.reshape(value, (8, 12, 64, 2)) @ torch.reshape(softmax(norm_S), (8, 12, 2, 2))
    #print(value_prime.shape)


    bs, num_att_heads, seq_len, attention_head_size = key.size()
    S = query @ key.transpose(-2, -1)
    S = S * (1.0 / math.sqrt(key.size(-1)))
    S = S + attention_mask
    norm_S = F.softmax(S, dim=-1)
    value_prime = norm_S @ value

    attn_value = value_prime.transpose(1, 2).contiguous().view(bs, seq_len, self.attention_head_size * self.num_attention_heads)


    # Concat multi-heads and recover the original shape
    # S is shaped: [2, 12, 64, 8]
    #bs, num_att_heads, seq_len, seq_len2 = S.shape
    #attn_value = torch.reshape(value_prime, (bs, seq_len, num_att_heads * self.attention_head_size))
    #print(attn_value)

    return attn_value
    #raise NotImplementedError


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    #print(hidden_states.shape)
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # multi-head attention
    self.self_attention = BertSelfAttention(config)
    # add-norm
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # another add-norm
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    this function is applied after the multi-head attention layer or the feed forward layer
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized 

    # first lets transform the output using the dense_layer
    ln_out = dense_layer(output)

    # next lets add this to the sub-layer input
    #   Ln shape: torch.Size([2, 64, 768])
    #   Input shape: torch.Size([2, 8, 768])

    norm_input = dropout(ln_out) + input
    # normalize the result
    final_norm = ln_layer(norm_input)

    return final_norm

    ##raise NotImplementedError


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the input and output of the multi-head attention layer
    3. a feed forward layer
    4. a add-norm that takes the input and output of the feed forward layer
    """
    

    att_output = self.self_attention.forward(hidden_states, attention_mask)
    add_output = self.add_norm(hidden_states, att_output, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
    feedfwd_output = self.interm_dense(add_output)
    feedfwd_output = self.interm_af(feedfwd_output)
    
    return self.add_norm(add_output, feedfwd_output, self.out_dense, self.out_dropout, self.out_layer_norm)
    #raise NotImplementedError



class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    #CGU from Junyang Lin et al., 2018
    self.relu = nn.ReLU()
    self.cgu_att = BertSelfAttention(config)
    self.cnn = self.post_embed_cnn = nn.Conv1d(8, 8, 2, padding=0, bias=True)




    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Get word embedding from self.word_embedding into input_embeds.
    ### TODO
    #print(input_ids)
    input_embeds = self.word_embedding(input_ids)
    # raise NotImplementedError


    # Get position index and position embedding from self.pos_embedding into pos_embeds.
    # position_ids was made 1D
    # indexes all rows up to seq_length columns
    pos_ids = self.position_ids[:, :seq_length]
    ### TODO
    # pos_embeds = self.pos_embedding[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)
    # raise NotImplementedError


    # Get token type ids, since we do not consider token type, just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    # includes word and position embeddings, the token types are negligible in calculation
    ### TODO
    embeds = input_embeds + pos_embeds + tk_type_embeds
    embeds = self.embed_layer_norm(embeds)
    embeds = self.embed_dropout(embeds)

    return embeds


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward_without_CGU(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    #CGU:
    x = self.cnn(sequence_output)
    unit = self.relu(sequence_output)
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
    unit = self.cgu_att.forward(unit, extended_attention_mask)

    #based on article, i get the impression we want to multiply
    sequence_output = sequence_output * unit

    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

"""
# The two classes bellow come mostly from https://github.com/IwasakiYuuki/Bert-abstractive-text-summarization/blob/master/transformer/Models.py
class Decoder(nn.Module):
  ''' A decoder model with self attention mechanism. '''

  def __init__(
          self,
          n_tgt_vocab, len_max_seq, d_word_vec,
          n_layers, n_head, d_k, d_v,
          d_model, d_inner, dropout=0.1):

    super().__init__()
    n_position = len_max_seq + 1

    self.tgt_word_emb = nn.Embedding(
      n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

    self.position_enc = nn.Embedding.from_pretrained(
      get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
      freeze=True)

    self.layer_stack = nn.ModuleList([
      DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
      for _ in range(n_layers)])

  def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

    dec_slf_attn_list, dec_enc_attn_list = [], []

    # -- Prepare masks
    non_pad_mask = get_non_pad_mask(tgt_seq)

    slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
    slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
    slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

    dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

    # -- Forward
    dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

    for dec_layer in self.layer_stack:
      dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
        dec_output, enc_output,
        non_pad_mask=non_pad_mask,
        slf_attn_mask=slf_attn_mask,
        dec_enc_attn_mask=dec_enc_attn_mask)

      if return_attns:
        dec_slf_attn_list += [dec_slf_attn]
        dec_enc_attn_list += [dec_enc_attn]

    if return_attns:
      return dec_output, dec_slf_attn_list, dec_enc_attn_list
    return dec_output,

class Transformer(nn.Module):
  ''' A sequence to sequence model with attention mechanism. '''

  def __init__(
          self,
          n_src_vocab, n_tgt_vocab, len_max_seq,
          d_word_vec=512, d_model=512, d_inner=2048,
          n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
          tgt_emb_prj_weight_sharing=True,
          emb_src_tgt_weight_sharing=True):

    super().__init__()

    #we want our encoder to be our BertModel, but how would we initalize it here?
    self.encoder = BertModel(config)
    '''(
    n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
    d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
    n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
    dropout=dropout) '''

    self.decoder = Decoder(
      n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
      d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
      n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
      dropout=dropout)

    self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
    nn.init.xavier_normal_(self.tgt_word_prj.weight)

    assert d_model == d_word_vec, \
      'To facilitate the residual connections, \
       the dimensions of all module outputs shall be the same.'

    if tgt_emb_prj_weight_sharing:
      # Share the weight matrix between target word embedding & the final logit dense layer
      self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
      self.x_logit_scale = (d_model ** -0.5)
    else:
      self.x_logit_scale = 1.

    if emb_src_tgt_weight_sharing:
      # Share the weight matrix between source & target word embeddings
      assert n_src_vocab == n_tgt_vocab, \
        "To share word embedding table, the vocabulary size of src/tgt shall be the same."
      self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

  def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

    tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

    enc_output, *_ = self.encoder(src_seq, src_pos)
    dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
    seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

    return seq_logit.view(-1, seq_logit.size(2))
    """
