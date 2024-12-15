#https://github.com/sooftware/attentions/blob/master/attentions.py

from collections import OrderedDict
from typing import Any, BinaryIO, ContextManager, Dict, List, Optional, Tuple
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, BertForQuestionAnswering

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss



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


class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class bert_model(BertForQuestionAnswering):
    def __init__(self, BertConfig):
        super().__init__(BertConfig)
        print("!!!!!!!!!!!!!!!!!!!   BertForQuestionAnswering   !!!!!!!!!!!!!!!!!!!")
        print("hidden_size: ",BertConfig.hidden_size)
        print("no of labels: ",BertConfig.num_labels)

        self.num_labels = BertConfig.num_labels

        self.bert = BertModel(BertConfig, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(BertConfig.hidden_size, BertConfig.num_labels)
        # self.multihead_attn = nn.MultiheadAttention(BertConfig.hidden_size, 8)
        # self.MLP = nn.Linear(2*BertConfig.hidden_size, BertConfig.hidden_size)
        # self.lii = nn.Linear(400,512)

        self.linear_layer = nn.Linear(BertConfig.hidden_size*3,BertConfig.hidden_size)

        # Attention Flow Layer
        self.att_weight_c = nn.Linear(768, 1)
        self.att_weight_q = nn.Linear(768, 1)
        self.att_weight_cq = nn.Linear(768, 1)


    # def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids = None, head_mask = None, inputs_embeds = None, start_positions = None, end_positions = None):
    def forward(
        self,
        kg_input_ids=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # return_dict = return_dict if return_dict is not None else self.BertConfig.use_return_dict
        

        
        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size )
            :param q: (batch, q_len, hidden_size )
            :return: (batch, c_len, q_len)
            """
            # print("c_len : ",c.shape)  #torch.Size([4, 512, 1024])
            # print("q_len : ",q.shape)  #torch.Size([4, 512, 1024])
            c_len = c.size(1)
            # print("c_len : ",c_len)  #512
            q_len = q.size(1)
            # print("q_len : ",q_len)  #512

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)
            # print("cq shape: ",cq.shape)  #torch.Size([4, 512, 512])

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq
            # print("s shape: ",s.shape)   #torch.Size([4, 512, 512])

            # (batch, c_len, q_len)

            a = F.softmax(s, dim=2)
            # print("a shape: ",a.shape)   #torch.Size([4, 512, 512])
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)

            c2q_att = torch.bmm(a, q)
            # print("c2q_att shape: ",c2q_att.shape)  #torch.Size([4, 512, 1024])
            # (batch, 1, c_len)

            # attn_output, attn_output_weights = self.multihead_attn(c2q_att, c2q_att, c2q_att)

            # b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # # print("b shape: ",b.shape)  # torch.Size([4, 1, 1024])
            # # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)

            # q2c_att = torch.bmm(b, c).squeeze()
            # # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 1024])
            # # (batch, c_len, hidden_size * 2) (tiled)

            # q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 512, 1024])
            # # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att], dim=-1)
            # print("x shape: ",x.shape)   #torch.Size([4, 512, 3072])
            return x

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # print("sequence_output: ", sequence_output.size()) # torch.Size([32, 512, 768]) bs,seq_len, 768

        kg_outputs = self.bert(kg_input_ids)
        kg_sequence_output = kg_outputs[0]

        # print("triple_emb: ", kg_sequence_output.size()) #triple_emb:  torch.Size([32, 100, 768]) bs,seq_len, 768


        
        #attention flow layer
        g = att_flow_layer(sequence_output, kg_sequence_output)
        # print("attention flow layer shape: ",g.shape)  # torch.Size([4, 512, 3072])

        join_output= self.linear_layer(g)
        # print("join_output: ",join_output.size())



        # kg_sequence_output_1 = torch.permute(kg_sequence_output,(0,2,1))
        # # print("kg_sequence_output_1: ", kg_sequence_output_1.size()) #triple_emb:  torch.Size([32, 768, 100]) bs,seq_len, 768

        # kg_sequence_output_2 = self.lii(kg_sequence_output_1)
        # # print("kg_sequence_output_2: ", kg_sequence_output_2.size()) #triple_emb:  torch.Size([32, 768, 512]) bs,seq_len, 768

        # kg_sequence_output_3 = torch.permute(kg_sequence_output_2,(0,2,1))
        # # print("kg_sequence_output_1: ", kg_sequence_output_3.size()) #triple_emb:  torch.Size([32, 512, 768]) bs,seq_len, 768

        # triple_cat=torch.cat((sequence_output,kg_sequence_output_3),2)
        # # print("triple_cat: ", triple_cat.size()) # triple_cat:  torch.Size([32, 512, 2*768])

        # triple_emd = self.MLP(triple_cat)
        # # print("triple_emd: ", triple_emd.size()) # triple_cat:  torch.Size([32, 512, 768])

        # # join_output= self.linear_layer(triple_cat)
        # # print("join_output: ", join_output.size()) # triple_cat:  torch.Size([32, 542, 768])

        logits = self.qa_outputs(join_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

