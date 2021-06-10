import torch.nn as nn
from src.utils.utils import *
import torch

class FARNN(nn.Module):
    def __init__(self, pretrained_embed=None, trans_r_1=None, trans_r_2=None, embed_r=None, trans_wildcard=None, config=None, h1=None):
        """
        Parameters
        ----------
        pretrained_embed: pretrained glove embedding,  V x D, numpy array
        trans_r_1: Tensor decomposition components 1, S x R (state x rank) numpy array
        trans_r_2: Tensor decomposition components 2, S x R (state x rank) numpy array
        embed_r: Tensor decomposition components 0, V x R (vocab size x R) numpy array
        config: config
        """
        super(FARNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embed).float(), freeze=(not bool(config.train_word_embed))) # V x D
        vocab_size, embed_dim = pretrained_embed.shape
        _, rank = embed_r.shape
        n_state, _ = trans_r_1.shape

        self.AS = config.additional_state
        self.V = vocab_size
        self.D = embed_dim
        self.S = n_state
        self.R = rank

        self.beta = config.beta
        self.bias_init = config.bias_init
        self.farnn = config.farnn
        self.random = bool(config.random)
        self.gate_activation = self.Sigmoidal()

        if self.farnn == 1:
            self.Wss1 = nn.Parameter((torch.randn((self.S + self.AS, self.S + self.AS))).float(), requires_grad=True)
            self.Wrs1 = nn.Parameter((torch.randn((self.R, self.S + self.AS))).float(), requires_grad=True)
            self.bs1 = nn.Parameter((torch.ones((1, self.S + self.AS)).float() * self.bias_init), requires_grad=True)

            self.Wss2 = nn.Parameter((torch.randn((self.S + self.AS, self.S + self.AS))).float(), requires_grad=True)
            self.Wrs2 = nn.Parameter((torch.randn((self.R, self.S + self.AS))).float(), requires_grad=True)
            self.bs2 = nn.Parameter((torch.ones((1, self.S + self.AS)).float() * self.bias_init), requires_grad=True)

            nn.init.xavier_normal_(self.Wss2)
            nn.init.xavier_normal_(self.Wrs2)
            nn.init.xavier_normal_(self.Wss1)
            nn.init.xavier_normal_(self.Wrs1)

        self.is_cuda = torch.cuda.is_available()
        self.h0 = self.hidden_init() # S hidden state dim should be equal to the state dim
        self.beta_vec = nn.Parameter(
            torch.tensor([self.beta] * self.R).float(), requires_grad=bool(config.train_beta)
        )

        if config.additional_nonlinear == 'tanh':
            self.additional_nonlinear = torch.nn.Tanh()
        elif config.additional_nonlinear == 'relu':
            self.additional_nonlinear = torch.nn.ReLU()
        elif config.additional_nonlinear == 'none':
            self.additional_nonlinear = lambda x: x
        else:
            raise NotImplementedError()

        if config.activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif config.activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif config.activation == 'none':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

        # initialize the wildcard matrix
        temp_wildcard = np.random.randn(self.S+self.AS, self.S+self.AS)*config.random_noise # very small noise
        temp_wildcard[:self.S, :self.S] = trans_wildcard
        self.trans_wildcard = nn.Parameter(
            torch.from_numpy(temp_wildcard).float(),
            requires_grad=bool(config.train_wildcard)
        ) # S x S

        self.embed_r = nn.Embedding.from_pretrained(torch.from_numpy(embed_r).float(), freeze=(not bool(config.train_V_embed))) # V x R
        self.trans_r_1 = nn.Parameter(
            torch.cat((torch.from_numpy(trans_r_1).float(), torch.randn((self.AS, self.R))*config.random_noise), dim=0),
            requires_grad=bool(config.train_fsa)
        ) # S x R
        self.trans_r_2 = nn.Parameter(
            torch.cat((torch.from_numpy(trans_r_2).float(), torch.randn((self.AS, self.R))*config.random_noise), dim=0),
            requires_grad=bool(config.train_fsa)
        ) # S x R
        embed_init_weight = self.embedding.weight.data # V x D
        _pinv = embed_init_weight.pinverse() # D x V
        _V = self.embed_r.weight.data # V x R
        self.embed_r_generalized = nn.Parameter(torch.matmul(_pinv, _V), requires_grad=bool(config.train_fsa)) # D x R

        if self.random: # random GRU, RNN
            self.initialize()

    def initialize(self):
        for params in self.parameters():
            try:
                nn.init.xavier_normal_(params)
            except:
                pass

    def Sigmoidal(self, exponent=1):
        def func(x):
            assert exponent > 0
            input = x * exponent
            return nn.functional.sigmoid(input)
        return func

    def forward(self, input, lengths):

        all_emb = self.embedding(input)
        all_regex_emb = self.embed_r(input) # B x L x R

        B, L, D = all_emb.size() # B x L x D

        self.batch_h0 = self.h0.unsqueeze(0).repeat(B, 1)
        hidden = self.h0.unsqueeze(0).repeat(B, 1) # B x S

        all_hidden = torch.zeros((B, L, self.S + self.AS)).cuda() if self.is_cuda else torch.zeros((B, L, self.S + self.AS))

        for i in range(L):
            emb = all_emb[:, i, :] # B x D

            L_generalized = torch.matmul(emb, self.embed_r_generalized) # B x D, D x R -> B x R
            L_generalized = self.additional_nonlinear(L_generalized)

            L_regex = all_regex_emb[:, i, :]
            L = L_regex * self.beta_vec + L_generalized * (1 - self.beta_vec)

            R = torch.matmul(hidden, self.trans_r_1)  # B x S, S x R -> B x R
            LR = torch.einsum('br,br->br', L, R)  # B x R, B x R -> B x R

            hidden_language = torch.matmul(LR, self.trans_r_2.transpose(0, 1)) # B x R, R x S  -> B x S
            hidden_wildcard = torch.matmul(hidden, self.trans_wildcard) # B x S, S x S -> B x S
            hidden_ = self.activation(hidden_language + hidden_wildcard)

            if self.farnn == 0:
                hidden = hidden_
            if self.farnn == 1:
                hidden = torch.einsum('bs,bs->bs', (1 - self.zt), hidden) + torch.einsum('bs,bs->bs', self.zt, hidden_)

            all_hidden[:, i, :] = hidden

        return all_hidden

    def maxmul(self, hidden, transition):
        temp = torch.einsum('bs,bsj->bsj', hidden, transition)
        max_val, _ = torch.max(temp, dim=1)
        return max_val

    def viterbi(self, input, lengths):
        """
        unbatched version of forward.
        input: Sequence of Vectors in one sentence, matrix in B x L x D
        lengths: lengths vector in B

        https://towardsdatascience.com/taming-lstms-variable-sized-mini-\
        batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        explains what is packed sequence
        need to deal with mask lengths
        :return all hidden state B x L x S:
        """

        all_emb = self.embedding(input)
        all_regex_emb = self.embed_r(input)  # B x L x R

        B, L, D = all_emb.size()  # B x L x D

        hidden = self.h0.unsqueeze(0).repeat(B, 1) # B x S

        all_hidden = torch.zeros((B, L, self.S + self.AS)).cuda() if self.is_cuda else torch.zeros((B, L, self.S + self.AS))

        for i in range(L):
            emb = all_emb[:, i, :]  # B x D

            L_generalized = torch.matmul(emb, self.embed_r_generalized)  # B x D, D x R -> B x R
            L_generalized = self.additional_nonlinear(L_generalized)

            L_regex = all_regex_emb[:, i, :]
            L = L_regex * self.beta_vec + L_generalized * (1 - self.beta_vec)

            Tr1 = torch.einsum('br,rs->brs', L, self.trans_r_1.transpose(1, 0))  # B x R, R x S -> B x R x S
            Tr = torch.einsum('sr,brj->bjs', self.trans_r_2, Tr1)  # S x R, B x R x S -> B x S x S
            Tr = Tr + self.trans_wildcard # B x S x S + S x S  -> B x S x S
            hidden_ = self.activation(self.maxmul(hidden, Tr))  # B x S,  B x S x S  -> B x S

            if self.farnn == 0:
                hidden = hidden_
            if self.farnn == 1:
                hidden = torch.einsum('bs,bs->bs', (1 - self.zt), hidden) + torch.einsum('bs,bs->bs', self.zt, hidden_)

            all_hidden[:, i, :] = hidden

        return all_hidden

    def hidden_init(self):
        hidden = torch.zeros((self.S + self.AS), dtype=torch.float)
        hidden[0] = 1.0
        hidden = hidden.cuda() if self.is_cuda else hidden
        return hidden


class IntentClassification(nn.Module):
    def __init__(self, pretrained_embed=None, trans_r_1=None, trans_r_2=None, embed_r=None, trans_wildcard=None, config=None,
                mat=None, bias=None):

        super(IntentClassification, self).__init__()

        self.fsa_rnn = FARNN(pretrained_embed=pretrained_embed,
                             trans_r_1=trans_r_1,
                             trans_r_2=trans_r_2,
                             embed_r=embed_r,
                             trans_wildcard=trans_wildcard,
                             config=config,)

        self.S, self.R = trans_r_1.shape
        self.V, self.D = pretrained_embed.shape
        self.clamp_score = bool(config.clamp_score)
        self.wfa_type = config.wfa_type

        if not config.random:
            self.mat = nn.Parameter(torch.cat([torch.from_numpy(mat).float(), torch.randn(config.additional_state, mat.shape[1])*config.random_noise]), requires_grad=bool(config.train_linear))
            self.bias = nn.Parameter(torch.from_numpy(bias).float(), requires_grad=bool(config.train_linear))
        else:
            self.mat = nn.Parameter(torch.randn(mat.shape[0]+config.additional_state, mat.shape[1]).float(), requires_grad=bool(config.train_linear))
            self.bias = nn.Parameter(torch.randn(bias.shape).float(), requires_grad=bool(config.train_linear))

    def forward(self, input, lengths):
        if self.wfa_type == 'forward':
            out = self.fsa_rnn.forward(input, lengths)
        elif self.wfa_type == 'viterbi':
            out = self.fsa_rnn.viterbi(input, lengths)
        else:
            raise NotImplementedError()

        B, L = input.size()
        last_hidden = out[torch.arange(B), lengths - 1, :]  # select all last hidden

        scores = torch.matmul(last_hidden, self.mat) + self.bias

        if self.clamp_score:
            scores = torch.clamp(scores, 0, 1)

        return scores  # B x C