import logging
import math
import random

import pickle
import json
from collections import OrderedDict
from typing import Any, BinaryIO, ContextManager, Dict, List, Optional, Tuple
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss


import torch
from torch import Tensor
from torch.nn import Parameter


vocab_path="../vocab/roberta/bioasq_entity_vocab_dic.pkl"
vocab = pickle.load(open(vocab_path,'rb'))

class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.
    <Tip warning={true}>
    You can't unpack a `ModelOutput` directly. Use the [`~file_utils.ModelOutput.to_tuple`] method to convert it to a
    tuple before.
    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class StaticAttention(nn.Module):
    def __init__(self, hidden_size, entity_vocab, t_embed):
        super().__init__()
        self.t_embed = t_embed
        self.entity_vocab = entity_vocab
        self.hidden_size = hidden_size
        self.entity_embedding = nn.Embedding(self.entity_vocab, t_embed)
        self.MLP = nn.Linear(3 * self.t_embed, 3 * self.t_embed)
        self.lii = nn.Linear(3*self.t_embed, self.hidden_size, bias=False)

    def forward(self, kg_enc_input):
        batch_size, _, _ = kg_enc_input.size()
        head, rel, tail = torch.split(kg_enc_input, 1, 2)  # (bsz, pl, tl)
        head_emb =  self.entity_embedding(head.squeeze(-1))  # (bsz, pl, tl, t_embed) 
        rel_emb = self.entity_embedding(rel.squeeze(-1)) # (bsz, pl, tl, t_embed)
        tail_emb = (self.entity_embedding(tail.squeeze(-1)))  # (bsz, pl, tl, t_embed)
        triple_cat =torch.cat([head_emb, rel_emb, tail_emb], 2)
        triple_emb = self.MLP(triple_cat)  # (bsz, pl, 3 * t_embed)
        triple_emb = self.lii(triple_emb)
        return triple_emb

class Trilinear_Att_layer(nn.Module):
    def __init__(self, hidden_size):
        super(Trilinear_Att_layer, self).__init__()
        print("!!!!!!!!!!!!!!!!! Trilinear_Att_layer !!!!!!!!!!!!")
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, 1)
        self.W2 = nn.Linear(hidden_size, 1) 
        self.W3 = nn.Parameter(torch.Tensor(1, 1, hidden_size))
        torch.nn.init.kaiming_uniform_(self.W3, a=math.sqrt(5))

    def forward(self, u, v):
        part1 = self.W1(u)     # batch * seq_len * 1
        part2 = self.W2(v).permute(0, 2, 1)   # batch * 1 * seq_len
        part3 = torch.bmm(self.W3*u, v.permute(0, 2, 1))  # batch * seq_len * seq_len
        # u_mask = (1.0 - u_mask.float()) * -10000.0
        # v_mask = (1.0 - v_mask.float()) * -10000.0
        # joint_mask = u_mask.unsqueeze(2) + v_mask.unsqueeze(1)    # batch * seq_len * num_paths
        # total_part = part1 + part2 + part3 + joint_mask
        total_part = part1 + part2 + part3
        return total_part

class OCN_Att_layer(nn.Module):
    def __init__(self, hidden_size):
        super(OCN_Att_layer, self).__init__()
        print("!!!!!!!!!!!!!!!!! OCN_Att_layer !!!!!!!!!!!!")
        self.hidden_size = hidden_size
        self.att = Trilinear_Att_layer(self.hidden_size)

    def forward(self, ol, ok):
        # print ('ol', ol.shape)
        # print ('ok', ok.shape)
        A = self.att(ol, ok)
        att = F.softmax(A, dim=1)    
        _OLK = torch.bmm(ol.permute(0, 2, 1), att).permute(0, 2, 1)       # batch *  hidden * seq_len
        OLK = torch.cat([ok-_OLK, ok*_OLK], dim=2)
        # print('OLK', OLK.shape)
        return OLK

class OCN_Merge_layer(nn.Module):
    def __init__(self, hidden_size):
        super(OCN_Merge_layer, self).__init__()
        print("!!!!!!!!!!!!!!!!! OCN_Merge_layer !!!!!!!!!!!!")
        self.hidden_size = hidden_size
        self.Wc = nn.Linear(self.hidden_size*12, self.hidden_size*6)
        self.Wg = nn.Linear(self.hidden_size*6, self.hidden_size)

    def forward(self, att_1, att_2, att_3, att_4, att_5, att_6):
        OCK = self.Wc(torch.cat([att_1, att_2, att_3, att_4, att_5, att_6], dim=2))   # batch * seq_len * hidden
        # print("OCK: ",OCK.shape)
        G = self.Wg(OCK)
        # print("G :",G.shape)
        return G

class OCN_CoAtt_layer(nn.Module):
    def __init__(self, hidden_size):
        super(OCN_CoAtt_layer, self).__init__()
        print("!!!!!!!!!!!!!!!!! OCN_CoAtt_layer !!!!!!!!!!!!")
        self.hidden_size = hidden_size
        self.att = Trilinear_Att_layer(self.hidden_size)
        self.Wp = nn.Linear(self.hidden_size*3, self.hidden_size)

    def forward(self, d, OCK):
        A = self.att(d, OCK)
        ACK = F.softmax(A, dim=2)    
        OA = torch.bmm(ACK, OCK)   
        APK = F.softmax(A, dim=1)
        POAA = torch.bmm(torch.cat([d, OA], dim=2).permute(0, 2, 1), APK).permute(0, 2, 1)
        OPK = F.relu(self.Wp(torch.cat([OCK, POAA], dim=2)))
        return OPK

class OCN_SelfAtt_layer(nn.Module):
    def __init__(self, hidden_size):
        super(OCN_SelfAtt_layer, self).__init__()
        print("!!!!!!!!!!!!!!!!! OCN_SelfAtt_layer !!!!!!!!!!!!")
        self.hidden_size = hidden_size
        self.att_sa = Trilinear_Att_layer(self.hidden_size)
        self.Wf = nn.Linear(self.hidden_size*4, self.hidden_size)

    def forward(self, OPK, _OPK):
        # A = self.att_sa(OPK, OPK_mask, _OPK, _OPK_mask)
        A = self.att_sa(OPK, _OPK)
        att = F.softmax(A, dim=1)    
        OSK = torch.bmm(OPK.permute(0, 2, 1), att).permute(0, 2, 1)      
        OFK = torch.cat([_OPK, OSK, _OPK-OSK, _OPK*OSK], dim=2)
        OFK = F.relu(self.Wf(OFK))
        return OFK

class fusion_layer(nn.Module):
    def __init__(self,config, hidden_size, vocab_size,  t_embed):
        super().__init__()
        print("!!!!!!!!!!!!!!!!! fusion_layer !!!!!!!!!!!!")
        self.config = config
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.t_embed = t_embed
        self.kg_attention = StaticAttention(self.config.hidden_size, self.vocab_size, self.t_embed)
        self.Rule_Att_layer = OCN_Att_layer(self.config.hidden_size)
        self.merge_layer = OCN_Merge_layer(self.config.hidden_size)
        self.co_att_layer = OCN_CoAtt_layer(self.config.hidden_size)
        self.self_attention_output = OCN_SelfAtt_layer(self.config.hidden_size)

    def forward(self,r1_encode,r2_encode,r3_encode,r4_encode,r5_encode,r6_encode,attention_output, roberta_output):
        r1_emb = self.kg_attention(r1_encode)
        r2_emb = self.kg_attention(r2_encode)
        r3_emb = self.kg_attention(r3_encode)
        r4_emb = self.kg_attention(r4_encode)
        r5_emb = self.kg_attention(r5_encode)
        r6_emb = self.kg_attention(r6_encode)
        att_1 = self.Rule_Att_layer(r1_emb,attention_output)
        att_2 = self.Rule_Att_layer(r2_emb,attention_output)
        att_3 = self.Rule_Att_layer(r3_emb,attention_output)
        att_4 = self.Rule_Att_layer(r4_emb,attention_output)
        att_5 = self.Rule_Att_layer(r5_emb,attention_output)
        att_6 = self.Rule_Att_layer(r6_emb,attention_output)
        merge_layer_output = self.merge_layer(att_1, att_2, att_3, att_4, att_5, att_6)
        co_att_layer_output = self.co_att_layer(roberta_output, merge_layer_output)
        self_attention_output = self.self_attention_output(merge_layer_output, merge_layer_output)
        return self_attention_output

class sequence_model(BertForSequenceClassification):
    def __init__(self, BertConfig):
        super().__init__(BertConfig)
        print("!!!!!!!!!!!!!!!!!!!   BertForSequenceClassification   !!!!!!!!!!!!!!!!!!!")
        self.bert = BertModel(BertConfig)
        classifier_dropout = (BertConfig.classifier_dropout if BertConfig.classifier_dropout is not None else BertConfig.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(BertConfig.hidden_size, 2)

    def forward(self, input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None,head_mask=None,inputs_embeds=None,labels=None,output_attentions=None,output_hidden_states=None,return_dict=None,):
        
        outputs = self.bert(input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

class QA_model(nn.Module):
    def __init__(self, model_name, config, vocab_size,  t_embed):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.vocab_size = vocab_size
        self.t_embed = t_embed
        print("hidden_size: ",self.config.hidden_size)
        print("vocab_size: ",vocab_size)
        print("t_embed: ",t_embed)
        self.text_emb = sequence_model.from_pretrained(model_name)
        self.kg_attention = StaticAttention(self.config.hidden_size, self.vocab_size, self.t_embed)
        self.fusion_layer = fusion_layer(self.config, self.config.hidden_size, self.vocab_size, self.t_embed)

        self.num_labels = self.config.num_labels
        print("!!!!!!!!!!!!!  num_labels: ",self.config.num_labels)
        print("!!!!!!!!!!!!!  hidden_size: ",self.config.hidden_size)

        print("!!!!!!!!!!!!!  KG_Injection_Att_layer !!!!!!!!!!!!")
        self.num_attention_heads = self.config.num_attention_heads
        print("!!!!!!!!!!!!!  num_attention_heads: ",self.num_attention_heads)
        self.attention_head_size = int(self.config.hidden_size / self.config.num_attention_heads)
        print("!!!!!!!!!!!!!  attention_head_size: ",self.attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        print("!!!!!!!!!!!!!  all_head_size: ",self.all_head_size)
        self.att_dropout = nn.Dropout(self.config.attention_probs_dropout_prob)
        print("!!!!!!!!!!!!!  att_dropout: ",self.att_dropout)
        self.output = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        print("!!!!!!!!!!!!!  output: ",self.output)
        self.RobertaLayerNorm = torch.nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        print("!!!!!!!!!!!!!  LayerNorm: ",self.RobertaLayerNorm)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        classifier_dropout = (self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob)
        self.class_dropout = nn.Dropout(classifier_dropout)

        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, kg_input,input_ids, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, 
                attention_mask=None, labels = None, token_type_ids=None, position_ids = None, 
                head_mask = None, inputs_embeds = None, return_dict=True):

        triple_emb = self.kg_attention(kg_input)
        outputs = self.text_emb(input_ids, attention_mask=attention_mask)
        pooled_sequence_output = outputs[1]
        sequence_output = pooled_sequence_output.unsqueeze(1).expand(-1, 512, -1)

        # print("#########  triple_emb size: ",triple_emb.size())
        # print("#########  sequence_output size: ",sequence_output.size())

        query_layer = self.transpose_for_scores(sequence_output)
        key_layer = self.transpose_for_scores(triple_emb)
        value_layer = self.transpose_for_scores(triple_emb)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.att_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        cs_attended = self.output(context_layer)
        cs_attended = self.dropout(cs_attended)
        attention_output = self.RobertaLayerNorm(cs_attended + sequence_output)
        # print("#########  cs_attended size: ",attention_output.size())
        join_output = self.fusion_layer(rule_1,rule_2,rule_3,rule_4,rule_5,rule_6,attention_output,sequence_output)
        # print("#########  rule size: ", join_output.size())
        join_output = self.class_dropout(join_output)
        logits = self.classifier(join_output)
        logits = logits.sum(dim=1)
        # print("#########  logits size: ", logits.size())
        # print(logits.view(-1, self.num_labels))
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

