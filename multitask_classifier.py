import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel, BertSelfAttention
from classifier import BertSentimentClassifier, SentimentDataset
from optimizer import AdamW
from tqdm import tqdm

from dfp_datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data
from dfp_utils import get_extended_attention_mask

from evaluation import model_eval_sst, test_model_multitask

#from transformer.Models import Decoder


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

# def grab_laysum(dataset):
#     test_data = load_data(args.test, 'test')
#     test_dataset = SentimentTestDataset(test_data, args)
#     test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

class MultitaskBERT(nn.Module):
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert parameters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        # we're going to need to load and process the larger dataset,
        # hopefully can use similar structures from bert.py
        # self.dataset = SentimentDataset(args.dataset)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.linear_paraphrase = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)


        #Abstractive Summarization Initalization from https://github.com/IwasakiYuuki/Bert-abstractive-text-summarization/blob/master/models.py

        self.encoder = self.bert
        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)
        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
        self.x_logit_scale = (d_model ** -0.5)
        self.o_l = nn.Linear(d_model, 512, bias=False)
        self.h_l = nn.Linear(512, 1, bias=True)
        nn.init.xavier_normal_(self.o_l.weight)
        nn.init.xavier_normal_(self.h_l.weight)
        self.a_l_1 = nn.Linear(d_model, 512, bias=False)
        self.a_l_2 = nn.Linear(d_model, 512, bias=False)
        nn.init.xavier_normal_(self.a_l_1.weight)
        nn.init.xavier_normal_(self.a_l_2.weight)


        #self.classifier = BertSentimentClassifier(config)

        self.relu = nn.ReLU()
        self.cgu_att = BertSelfAttention(config)
        self.post_embed_cnn = nn.Conv1d(embed_size, embed_size, 2, padding='same')


        #self.classifier = BertSentimentClassifier(config)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO

        # embeds = self.bert.embed(input_ids)

        #need to use bert forward
        pooler = self.bert.forward(input_ids, attention_mask)['pooler_output']
        return pooler

    """
    def forward_conv(self, input_ids, attention_mask):
      " " "
      input_ids: [batch_size, seq_len], seq_len is the max length of the batch
      attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
      " " "
      # get the embedding for each input token
      embedding_output = self.embed(input_ids=input_ids)

      # feed to a transformer (a stack of BertLayers)
      sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

      #CGU:
      # unit = self.relu(sequence_output)
      # extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
      # unit = self.cgu_att.forward(unit, extended_attention_mask)
      unit = self.post_embed_cnn(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE, 2, padding = 'same')

      #based on article, i get the impression we want to multiply
      sequence_output = sequence_output * unit

      # get cls token hidden state
      first_tk = sequence_output[:, 0]
      first_tk = self.pooler_dense(first_tk)
      first_tk = self.pooler_af(first_tk)

      return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
      """


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO

        #logits = self.classifier.forward(input_ids, attention_mask)

        pooler = self.forward(input_ids, attention_mask)
        logits = self.dropout(pooler)
        logits = self.linear(logits)

        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO

        log1 = self.forward(input_ids_1, attention_mask_1)
        log2 = self.forward(input_ids_2, attention_mask_2)
        concat_logs = torch.cat((log1, log2), dim=1)
        logits = self.linear_paraphrase(concat_logs)

        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO

        log1 = self.forward(input_ids_1, attention_mask_1)
        log2 = self.forward(input_ids_2, attention_mask_2)

        cosine_sim = torch.nn.functional.cosine_similarity(log1, log2)

        return cosine_sim
        #raise NotImplementedError

"""
def abs_summarization(self, input_ids, attention_mask, src_seq, tgt_seq, tgt_pos):
    #code mostly from https://github.com/IwasakiYuuki/Bert-abstractive-text-summarization/blob/master/models.py
    tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

    #unlike the github repo, our encoder takes in input ids and attention mask
    enc_output, _ = self.encoder.forward(input_ids, attention_mask)['pooler_output']
    dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

    #        o = self.o_l(dec_output)
    #        p_gen = torch.sigmoid(self.h_l(o).view(-1, 1))

    seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale
    #        a = self.a_l_1(dec_output)
    #        a = torch.bmm(a, enc_output)
    #        a = self.a_l_2(a)

    return seq_logit.view(-1, seq_logit.size(2))
"""



def save_model(model, optimizer, args, config, filepath):
save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'model_config': config,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
}

torch.save(save_info, filepath)
print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
# Load data
# Create the data and its corresponding datasets and dataloader
sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

sst_train_data = SentenceClassificationDataset(sst_train_data, args)
sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=sst_train_data.collate_fn)
sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                collate_fn=sst_dev_data.collate_fn)

# Init model
config = {'hidden_dropout_prob': args.hidden_dropout_prob,
          'num_labels': num_labels,
          'hidden_size': 768,
          'data_dir': '.',
          'option': args.option}

config = SimpleNamespace(**config)

model = MultitaskBERT(config)
model = model.to(device)

lr = args.lr
optimizer = AdamW(model.parameters(), lr=lr)
best_dev_acc = 0

# Run for the specified number of epochs
for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask, b_labels = (batch['token_ids'],
                                   batch['attention_mask'], batch['labels'])

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        optimizer.zero_grad()
        logits = model.predict_sentiment(b_ids, b_mask)
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    train_loss = train_loss / (num_batches)

    train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
    dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        save_model(model, optimizer, args, config, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")



def test_model(args):
with torch.no_grad():
    #sss
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath)
    config = saved['model_config']

    model = MultitaskBERT(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"Loaded model to test from {args.filepath}")

    test_model_multitask(args, model, device)


def get_args():
parser = argparse.ArgumentParser()
parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

parser.add_argument("--seed", type=int, default=11711)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--option", type=str,
                    help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                    choices=('pretrain', 'finetune'), default="pretrain")
parser.add_argument("--use_gpu", action='store_true')

parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

# hyper parameters
parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                    default=1e-5)

args = parser.parse_args()
return args

if __name__ == "__main__":
start = time.time()
args = get_args()
args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
seed_everything(args.seed)  # fix the seed for reproducibility
train_multitask(args)
test_model(args)
print("Total time elapsed when training on SST: {:.2f}s".format(time.time() - start))
