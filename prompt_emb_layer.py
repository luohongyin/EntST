import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    BertConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings, BertLayer
)

class PromptEmbedding(nn.Embedding):
    
    def __init__(self, embed_tokens, n_prompt, prompt_path=None):
        
        n_tok, n_dim = embed_tokens.weight.data.size()
        super(PromptEmbedding, self).__init__(n_tok, n_dim)

        self.embed_tokens = embed_tokens
        self.weight = embed_tokens.weight
        self.n_prompt = n_prompt
        
        if prompt_path is None:
            prompt_emb = torch.rand(n_prompt, n_dim)
        else:
            prompt_emb = torch.load(prompt_path).cpu()

        std = embed_tokens.weight.data.std()
        mean = embed_tokens.weight.data.mean()
        
        prompt_emb.normal_(mean, std)
        self.embed_tokens.weight.data = torch.cat(
            [self.embed_tokens.weight.data, prompt_emb],
            dim = 0
        )
        self.prompt_emb = nn.Parameter(prompt_emb)
    
    def forward(self, input_ids):
        cur_bs, seq_len = input_ids.size()
        
        p_emb = self.prompt_emb.unsqueeze(0).repeat(cur_bs, 1, 1)
        input_emb = self.embed_tokens(input_ids)

        cls_emb = input_emb[:, :1, :]
        txt_emb = input_emb[:, 1 + self.n_prompt:, :]
        
        output_emb = torch.cat(
            [
                cls_emb, p_emb, txt_emb
            ], dim=1
        )

        return output_emb


class PromptDecoder(nn.Module):

    def __init__(self, linear, n_prompt, prompt_emb):
        super(PromptDecoder, self).__init__()
        self.linear = linear
        self.weight = linear.weight
        self.n_prompt = n_prompt
        self.prompt_emb = prompt_emb

        self.linear.bias.data = torch.cat(
            [
                self.linear.bias.data,
                torch.zeros(self.n_prompt)
            ], dim = 0
        )
    
    def forward(self, x):
        linear_output = self.linear(x)
        cur_bs, seq_len, _ = linear_output.size()
        prompt_output = torch.ones(cur_bs, seq_len, self.n_prompt).cuda() * -1000
        output = torch.cat(
            [
                linear_output[:, :, :-self.n_prompt],
                prompt_output
            ], dim = 2
        )
        return output


class TuringAdaptorSCModel(nn.Module):

    def __init__(self, tok_path=None, model_path=None, num_cls=2):
        super(TuringAdaptorSCModel, self).__init__()
        self.config = BertConfig(
            vocab_size = 24,
            max_position_embeddings = 24,
            hidden_size = 1024,
            type_vocab_size = 2,
            layer_norm_eps = 1e-12,
            hidden_dropout_prob = 0.1,
            num_attention_heads = 8
        )
        # self.bert_embeddings = BertEmbeddings(self.config)
        self.bert_layer = BertLayer(self.config)
        self.tok = AutoTokenizer.from_pretrained(
            tok_path
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.positional_emb = nn.Embedding(
            self.num_hidden_layers, self.model.config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(
            self.config.hidden_size, eps=self.config.layer_norm_eps
        )
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.linear = nn.Linear(
            self.config.hidden_size, num_cls, bias=True
        )

    def get_model_outputs(self, input_ids, attn_mask):
        with torch.no_grad():
            results = self.model(
                input_ids = input_ids,
                attention_mask = attn_mask,
                output_hidden_states = True
            )
        
        cls_states = torch.cat([
            x[:, :1, :] for x in results.hidden_states[1:]
        ], dim = 1)
        batch_size, num_layer, _ = cls_states.size()
        
        pos_ids = torch.LongTensor([
            list(range(num_layer)) for x in range(batch_size)
        ])
        
        pos_ids = pos_ids.cuda()
        return cls_states, pos_ids
    
    def forward(
            self, input_ids = None,
            attention_mask = None, labels = None
        ):
        loss_fn = nn.CrossEntropyLoss()
        cls_states, pos_ids = self.get_model_outputs(
            input_ids, attention_mask
        )
        pos_emb = self.positional_emb(pos_ids)
        
        input_emb = self.LayerNorm(cls_states + pos_emb)
        input_emb = self.dropout(input_emb)

        output_emb = self.bert_layer(input_emb)[0][:, -1, :]
        logits = self.linear(output_emb)
        if labels is not None:
            loss = loss_fn(logits, labels)
        else:
            loss = None
        # print(output_emb.size())
        # print(logits.size())
        # abort()
        return logits, loss


def encode_inputs(self, tok, input_txt, device):
    input_enc = tok(
        text = input_txt,
        max_length = 512,
        padding = 'longest',
        return_tensors = 'pt',
        truncation = True,
        return_attention_mask = True,
        verbose = False
    )

    input_ids = input_enc.input_ids
    attn_mask = input_enc.attention_mask
    
    if device != torch.device('cpu'):
        input_ids = input_ids.cuda()
        attn_mask = attn_mask.cuda()
    return input_ids, attn_mask