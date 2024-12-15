from collections import OrderedDict
from typing import Any, BinaryIO, ContextManager, Dict, List, Optional, Tuple
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, RobertaForQuestionAnswering

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from torch_geometric.data import Data

import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)

# import torch_scatter


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

# class StaticAttention(nn.Module):
#     def __init__(self, hidden_size, vocab_dic, t_embed):
#         super().__init__()
#         self.t_embed = t_embed
#         self.vocab_dic = vocab_dic
#         # self.relation_vocab = relation_vocab
#         self.hidden_size = hidden_size
#         self.vocab_embedding = nn.Embedding(self.vocab_dic, t_embed)
#         # self.entity_embedding.weight = nn.Parameter(embedding_matrix_entity, requires_grad=True)
#         # self.rel_embedding = nn.Embedding(self.relation_vocab, t_embed)
#         # self.rel_embedding.weight = nn.Parameter(embedding_matrix_rel, requires_grad=True)
#         self.MLP = nn.Linear(3 * self.t_embed, 3 * self.t_embed)
#         # self.lii = nn.Linear(3*self.t_embed, self.hidden_size, bias=False)
#         self.lii = nn.Linear(self.t_embed, self.hidden_size, bias=False)
#         self.linear = nn.Linear(self.t_embed,self.t_embed)

#     def forward(self, kg_enc_input):
#         print("kg_enc_input size: ",kg_enc_input.size()) #torch.Size([8, 512, 3])
#         batch_size, _, _ = kg_enc_input.size()
#         # print("batch_size :",batch_size)
#         head, rel, tail = torch.split(kg_enc_input, 1, 2)  # (bsz, pl, tl)
#         # print("head shape: ",head.shape) #torch.Size([bsz, 512, 1])
#         # print("rel shape: ",rel.shape) #torch.Size([bsz, 512, 1])
#         # print("tail shape: ",tail.shape) #torch.Size([bsz, 512, 1])
#         head_emb =  self.vocab_embedding(head.squeeze(-1))  # (bsz, pl, tl, t_embed) 
#         # print("head_emb shape: ",head_emb.shape) #torch.Size([bsz, 512, 300])
#         rel_emb = self.vocab_embedding(rel.squeeze(-1)) # (bsz, pl, tl, t_embed)
#         # print("rel_emb shape: ",rel_emb.shape) #torch.Size([bsz, 512, 300])
#         tail_emb = (self.vocab_embedding(tail.squeeze(-1)))  # (bsz, pl, tl, t_embed)
#         # print("tail_emb shape: ",tail_emb.shape) #torch.Size([bsz, 512, 300])

#         # triple_cat =torch.cat([head_emb, rel_emb, tail_emb], 2)
#         # # print("triple_cat shape: ",triple_cat.shape)
#         # triple_emb = self.MLP(triple_cat)  # (bsz, pl, 3 * t_embed)
#         # # print("triple_emb shape: ",triple_emb.shape) #torch.Size([bsz, 512, 900])
#         # triple_emb = self.lii(triple_emb)
#         # # print("triple_emb shape after linear layer: ",triple_emb.shape) #torch.Size([bsz, 512, 768])
#         # return triple_emb

#         tail_dash= self.linear(head_emb) - self.linear(rel_emb)
#         # print("tail_dash shape: ",tail_dash.shape) 
#         characters = torch.max(tail_emb, 1)[1]
#         # print("characters shape: ",characters.shape)
#         loss_emb = nn.CrossEntropyLoss()
#         emb_loss=loss_emb(tail_dash,characters)
#         triple_emb = self.lii(tail_dash)
#         # print("triple_emb shape after linear layer: ",triple_emb.shape) #torch.Size([bsz, 512, 768])
#         return triple_emb,emb_loss

# class roberta_model(RobertaForQuestionAnswering):
#     def __init__(self, RobertaConfig):
#         super().__init__(RobertaConfig)
#         print("!!!!!!!!!!!!!!!!!!!   RobertaForQuestionAnswering   !!!!!!!!!!!!!!!!!!!")

#         self.num_labels = RobertaConfig.num_labels
#         # print("!!!!!!!!!!!!!:  num_labels: ",RobertaConfig.num_labels)
#         # print("!!!!!!!!!!!!!:  hidden_size: ",RobertaConfig.hidden_size)

#         self.linear_layer = nn.Linear(4096, 1024)

#         # Attention Flow Layer
#         self.att_weight_c = nn.Linear(1024, 1)
#         self.att_weight_q = nn.Linear(1024, 1)
#         self.att_weight_cq = nn.Linear(1024, 1)

#         # self.roberta = RobertaModel(RobertaConfig, add_pooling_layer=False)
#         self.qa_outputs = nn.Linear(RobertaConfig.hidden_size, RobertaConfig.num_labels)

#     # def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids = None, head_mask = None, inputs_embeds = None, start_positions = None, end_positions = None):
#     def forward(
#         self,
#         triple_emb=None,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         start_positions=None,
#         end_positions=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         # return_dict = return_dict if return_dict is not None else self.BertConfig.use_return_dict
        
#         def att_flow_layer(c, q):
#             """
#             :param c: (batch, c_len, hidden_size )
#             :param q: (batch, q_len, hidden_size )
#             :return: (batch, c_len, q_len)
#             """
#             # print("c_len : ",c.shape)  #torch.Size([4, 512, 768])
#             # print("q_len : ",q.shape)  #torch.Size([4, 512, 768])
#             c_len = c.size(1)
#             # print("c_len : ",c_len)  #512
#             q_len = q.size(1)
#             # print("q_len : ",q_len)  #512

#             cq = []
#             for i in range(q_len):
#                 #(batch, 1, hidden_size * 2)
#                 qi = q.select(1, i).unsqueeze(1)
#                 #(batch, c_len, 1)
#                 ci = self.att_weight_cq(c * qi).squeeze()
#                 cq.append(ci)
#             # (batch, c_len, q_len)
#             cq = torch.stack(cq, dim=-1)
#             # print("cq shape: ",cq.shape)  #torch.Size([4, 512, 512])

#             # (batch, c_len, q_len)
#             s = self.att_weight_c(c).expand(-1, -1, q_len) + \
#                 self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
#                 cq
#             # print("s shape: ",s.shape)   #torch.Size([4, 512, 512])

#             # (batch, c_len, q_len)

#             a = F.softmax(s, dim=2)
#             # print("a shape: ",a.shape)   #torch.Size([4, 512, 512])
#             # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)

#             c2q_att = torch.bmm(a, q)
#             # print("c2q_att shape: ",c2q_att.shape)  #torch.Size([4, 512, 768])
#             # (batch, 1, c_len)

#             b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
#             # print("b shape: ",b.shape)  # torch.Size([4, 1, 512])
#             # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)

#             q2c_att = torch.bmm(b, c).squeeze()
#             # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 768])
#             # (batch, c_len, hidden_size * 2) (tiled)

#             q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
#             # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 512, 768])
#             # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

#             # (batch, c_len, hidden_size * 8)
#             x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
#             # print("x shape: ",x.shape)   #torch.Size([4, 512, 3072])
#             return x



#         # outputs = self.roberta(
#         #     input_ids,
#         #     attention_mask=attention_mask,
#         #     token_type_ids=token_type_ids,
#         #     position_ids=position_ids,
#         #     head_mask=head_mask,
#         #     inputs_embeds=inputs_embeds,
#         #     output_attentions=output_attentions,
#         #     output_hidden_states=output_hidden_states,
#         #     return_dict=return_dict,
#         # )

#         # sequence_output = outputs[0]

#         # print("sequence_output: ", sequence_output.size()) # batch_size, sequence_length, hidden_size(1024)
#         # print("triple_emb: ", triple_emb.size()) #triple_emb:  torch.Size([6, 512, 900]) # batch_size, sequence_len, 900
        
#         #attention flow layer
#         g = att_flow_layer(sequence_output, triple_emb)
#         # print("attention flow layer shape: ",g.shape)  # torch.Size([4, 512, 3072])

#         join_output= self.linear_layer(g)
#         # print("join_output: ",join_output.size())

#         logits = self.qa_outputs(join_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()

#         total_loss = None
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions = start_positions.clamp(0, ignored_index)
#             end_positions = end_positions.clamp(0, ignored_index)

#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2


#         if not return_dict:
#             # print("hhhhhhhhhhhhhhhhhhhh")
#             output = (start_logits, end_logits) + outputs[2:]
#             return (total_loss,) + output

#         #     if total_loss is not None:
#         #         print("llllllllllllllllllllllllll")
#         #         return (total_loss,) + output
#         #     else:
#         #         print("ppppppppppppppppp")
#         #         print(len(output))
#         #         return output
#         #     # return ((total_loss,) + output) if total_loss is not None else output
#         # print("kkkkkkkkkkkkkk")
#         # return total_loss,start_logits,end_logits,outputs.hidden_states,outputs.attentions

class GAT(torch.nn.Module): # optimized library implementation of gcn
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        # use our gat message passing
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4)
        self.conv2 = GATv2Conv(4 * hidden_dim, hidden_dim, heads=4)

        self.post_mp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(0.6),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index, adj=None):
        x = torch.clone(x.detach())
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=0.6, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=0.6, training=self.training)
        # MLP output
        x = self.post_mp(x)
        return x



class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True,
                 directed=False, self_loop=True, **kwargs):
        """
        Initialize a GCN layer.
        Args:
            in_channels      In-channel dimension of node embeddings
            out_channels     Out-channel dimension of node embeddings
            bias             A boolean value determining whether we add a
                                learnable bias term in linear transformation
            directed         A boolean value determining whether we use directed
                                message passing D^{-1}A or use symmetric normalized
                                adjacency matrix D^{-1/2}AD^{-1/2}
            self_loop        A boolean value determining whether we add a self-
                                loop for each node
        """
        super(GCNLayer, self).__init__(**kwargs, aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.directed = directed
        self.self_loop = self_loop

        # Define the layers needed for the message and update functions below.
        # self.lin is the linear transformation that we apply to the embedding.
        self.lin = nn.Linear(self.in_channels, self.out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset all learnable parameters in the linear transformation.
        """
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        """
        Produce a forward propagation of GCN layer.

        Args:
            x             The node embedding
            edge_index    The (2, |E|) adjacency list of the graph
            edge_weight   The (|E|) vector specifying the edge weights in the graph
                            (for unweighted graph, edge weight is 1)

        Returns:
            An updated node embedding
        """
        # Add self-loops to the adjacency matrix.
        if self.self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_weight = torch.cat((edge_weight, torch.ones(x.size(0))), dim=-1)

        # Apply linear transformation on node features.
        x = self.lin(x)

        # Compute normalization by updated node degree.
        if self.directed:
            row, _ = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)  # only out-degree
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0
            norm = deg_inv[row]
        else:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=(x, x), norm=norm, edge_weight=edge_weight)

    def message(self, x_j, edge_weight, norm):
        """
        Send the message of the neighboring node (i.e., x_j) to the source node (i.e., x_i).

        Args:
            x_j           The embedding of the neighboring node of source node x_i
            edge_weight   The edge weight of certain edge
            norm          Normalization constant determined by self.directed

        Returns:
            A message sending from the neighboring node to the source node
        """
        a = norm.view(-1, 1).to(torch.device("cuda"))
        b = edge_weight.view(-1, 1).to(torch.device("cuda"))
        w = a * x_j * b
        z = w.to(torch.device("cuda"))
        return z


class GCN(torch.nn.Module): ## Main gcn function
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        """
        Initialize a GCN model.
        Args:
            input_dim       Input dimension of node embeddings
            hidden_dim      Hidden dimension of node embeddings
            output_dim      Output dimension of node embeddings
            num_layers      The number of GCN layers
            dropout         The dropout ratio in (0, 1]
                              (dropout: the probability of an element getting zeroed)
            return_embeds   A boolean value determining whether we skip the
                              classification layer and return node embeddings
        """

        super(GCN, self).__init__()

        # Construct all convs
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, directed=False)
                                          for i in range(self.num_layers - 1)])

        # Construct batch normalization
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim)
                                        for i in range(self.num_layers - 1)])
        # First GCN layer
        self.convs[0] = GCNLayer(input_dim, hidden_dim, directed=False)
        # Last GCN layer
        self.last_conv = GCNLayer(hidden_dim, output_dim, directed=False)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        """
        Reset all learnable parameters in GCN layers and Batch Normalization
        Layers.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        """
        Produce a forward propagation of GCN model. Before the last GCN layer,
        we transform the embedding (x) in the following sequence:
          x -> GCN_Layer -> Batch_Norm -> ReLU -> Dropout.
        At the last GCN layer, the following sequence is applied:
          x -> GCN Layer -> Softmax -> output.

        Args:
            x             The node embedding
            edge_index    The adjacency list of the graph

        Returns:
            out           The predictions of labels / the updated node embedding
        """
        x = torch.clone(x.detach())
        for l in range(self.num_layers - 1):
            # Unweighted graph has weight 1.
            x = self.convs[l](x, edge_index, torch.ones(edge_index.shape[1]))
            x = self.bns[l](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.last_conv(x, edge_index, torch.ones(edge_index.shape[1]))
        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)

        return out


class GNN(nn.Module): # gnn class for gcn, gat etc...
    def __init__(self, hidden_size, vocab_size, t_embed=300):
        super().__init__()
        self.t_embed = t_embed
        self.hidden_size = hidden_size
        self.entity_embedding = nn.Embedding(vocab_size, self.t_embed) # vocab_size X t_embed ### embeddings of the entities of knowledge graph

        liss = []
        for i in range(vocab_size):
            liss.append(i)
        temp_lis = torch.tensor(liss, dtype=torch.long)
        self.embeddings = self.entity_embedding(temp_lis)  ### vocab_size X t_embed

        """
        For Bio-bert embeddings
        x = json.load(open("/home/harsh_1901cs23/Abstractive_Qa/embeddings.json"))
        listt = []
        for i in range(len(x.keys())):
            listt.append([])
        for k in x.keys():
            listt[int(k)] = x[k]
        self.embeddings = torch.tensor(listt).to(torch.device("cuda"))
        """

        #self.gcn = GCN(input_dim=t_embed, hidden_dim=t_embed, output_dim=t_embed, num_layers=2, dropout=0.5,
        #          return_embeds=True)  !!!!!!!!! Use this line to use gcn
        self.gcn = GAT(t_embed, t_embed, t_embed) ## change graph encoder here
        self.lii = nn.Linear(self.t_embed*3, self.hidden_size, bias=False)

    def forward(self, kg_enc_input):
        # processing for gcn, gat done here for every sample of batch separately
        print("GNN forward\n",kg_enc_input)
        kg = torch.split(kg_enc_input, dim=0, split_size_or_sections=1) # here every sample in batch is separated, if batch is (4, 60, 512) then
        output = []
        for kgi in kg:
            kg_inp = kgi.squeeze(0)
            head, rel, tail = torch.split(kg_inp, dim=1, split_size_or_sections=1)
            head = head.squeeze(-1)
            rel = rel.squeeze(-1)
            tail = tail.squeeze(-1)
            edge_list_1 = torch.stack([head, rel])
            edge_list_2 = torch.stack([rel, tail])

            edge_list = torch.cat([edge_list_1, edge_list_2], 1)
            # input is embeddings: vocab_size X t_embed
            #          edge_index: []
            print(edge_list.size())
            w = self.gcn.forward(x=self.embeddings, edge_index=edge_list)
            print("output dim of gat: ", w.size())
            #w = self.embeddings
            #w = self.gcn.forward((self.embeddings,edge_list))[0]

            # GCN

            triples_emb = []
            for row in kg_inp:
                triples_emb.append(torch.cat([w[row[0]], w[row[1]], w[row[2]]], 0))
            out_w = torch.stack(triples_emb)
            print("output after stacking as hrt:", out_w.size())
            output.append(out_w)


        final = torch.stack(output) ### combining all batches
        triple_emb = self.lii(final) ## matching dimensions
        return triple_emb



class QA_model(nn.Module):
    def __init__(self, model, hidden_size, vocab_size,  t_embed):
        super().__init__()

        self.model = model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.t_embed = t_embed
        print("hidden_size: ",hidden_size)
        print("vocab_size: ",vocab_size)
        print("t_embed: ",t_embed)

        # self.roberta_model= roberta_model.from_pretrained(model)
        # self.StaticAttention = StaticAttention(self.hidden_size, self.vocab_size, self.t_embed)
        self.gnn = GNN(self.hidden_size, self.vocab_size, self.t_embed)

    def forward(self, kg_input,
                input_ids, attention_mask=None, token_type_ids=None, position_ids = None, 
                head_mask = None, inputs_embeds = None, start_positions = None, end_positions = None):

        # print("!!!!!!!!!!!!!! kg_input  !!!!!!!!!!")
        # print(kg_input)
        # print(kg_input.size())

        triple_emb = self.gnn(kg_input)
        # print("triple_emb size: ",triple_emb.size())

        # loss,start_logits,end_logits = self.roberta_model(triple_emb, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, start_positions, end_positions)

        # if loss==None:
        #     final_loss= triple_loss
        # else:
        #     final_loss=triple_loss+loss

        # return QuestionAnsweringModelOutput(
        #     loss=final_loss,
        #     start_logits=start_logits,
        #     end_logits=end_logits,
        # )





