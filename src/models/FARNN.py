import torch.nn as nn
from src.utils.utils import *

class FSARNNIntegrateEmptyStateSaperateGRU(nn.Module):
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
        super(FSARNNIntegrateEmptyStateSaperateGRU, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embed).float(), freeze=(not bool(config.train_word_embed))) # V x D
        vocab_size, embed_dim = pretrained_embed.shape
        _, rank = embed_r.shape
        n_state, _ = trans_r_1.shape

        self.AS = config.additional_state
        self.V = vocab_size
        self.D = embed_dim
        self.S = n_state
        self.R = rank

        self.clip_neg = bool(config.clip_neg)
        self.activation = config.activation
        self.beta = config.beta
        self.bias_init = config.bias_init
        self.farnn = config.farnn
        self.xavier = config.xavier
        self.random = bool(config.random)

        self.gate_activation = self.Sigmoidal(config.sigmoid_exponent)

        if self.farnn == 1:
            self.Wss1 = nn.Parameter((torch.randn((self.S + self.AS, self.S + self.AS))).float(), requires_grad=True)
            self.Wrs1 = nn.Parameter((torch.randn((self.R, self.S + self.AS))).float(), requires_grad=True)
            self.bs1 = nn.Parameter((torch.ones((1, self.S + self.AS)).float() * self.bias_init), requires_grad=True)

            self.Wss2 = nn.Parameter((torch.randn((self.S + self.AS, self.S + self.AS))).float(), requires_grad=True)
            self.Wrs2 = nn.Parameter((torch.randn((self.R, self.S + self.AS))).float(), requires_grad=True)
            self.bs2 = nn.Parameter((torch.ones((1, self.S + self.AS)).float() * self.bias_init), requires_grad=True)
            if self.xavier:
                nn.init.xavier_normal_(self.Wss2)
                nn.init.xavier_normal_(self.Wrs2)
                nn.init.xavier_normal_(self.Wss1)
                nn.init.xavier_normal_(self.Wrs1)

        self.is_cuda = torch.cuda.is_available()
        self.h0 = self.hidden_init() # S hidden state dim should be equal to the state dim
        if self.farnn == 1:
            self.h1 = h1.cuda() if self.is_cuda else h1

        self.beta_vec = nn.Parameter(
            torch.tensor([self.beta] * self.R).float(), requires_grad=bool(config.train_beta)
        )
        self.additional_nonlinear = config.additional_nonlinear
        assert self.additional_nonlinear in ['tanh', 'sigmoid', 'relu', 'none']

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

    def Sigmoidal(self, exponent):
        def func(x):
            assert exponent > 0
            input = x * exponent
            return nn.functional.sigmoid(input)
        return func

    def cal_hidden_bar(self, hidden, L):
        if self.farnn == 0:  # 0 - FSARNN, 8 random FSARNN
            hidden_bar = hidden

        if self.farnn == 1:
            self.zt = self.gate_activation(torch.matmul(hidden, self.Wss1) + torch.matmul(L, self.Wrs1) + self.bs1)  # B x S
            rt = self.gate_activation(torch.matmul(hidden, self.Wss2) + torch.matmul(L, self.Wrs2) + self.bs2)  # B x S
            hidden_bar = torch.einsum('bs,bs->bs', (1 - rt), self.batch_h1) + torch.einsum('bs,bs->bs', rt,
                                                                                           hidden)  # reset to previous state
        return hidden_bar

    def forward(self, input, lengths):
        # TODO: length not used now
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
        all_regex_emb = self.embed_r(input) # B x L x R

        B, L, D = all_emb.size() # B x L x D

        if self.farnn == 1:
            self.batch_h1  = self.h1.unsqueeze(0).repeat(B, 1)  # B x S

        self.batch_h0 = self.h0.unsqueeze(0).repeat(B, 1)
        hidden = self.h0.unsqueeze(0).repeat(B, 1) # B x S

        all_hidden = torch.zeros((B, L, self.S + self.AS)).cuda() if self.is_cuda else torch.zeros((B, L, self.S + self.AS))

        for i in range(L):
            emb = all_emb[:, i, :] # B x D

            L_generalized = torch.matmul(emb, self.embed_r_generalized) # B x D, D x R -> B x R

            if self.additional_nonlinear == 'relu':
                L_generalized = nn.functional.relu(L_generalized)
            elif self.additional_nonlinear == 'tanh':
                L_generalized = torch.tanh(L_generalized)
            elif self.additional_nonlinear == 'sigmoid':
                L_generalized = torch.sigmoid(L_generalized)
            else:
                pass

            L_regex = all_regex_emb[:, i, :]
            L = L_regex * self.beta_vec + L_generalized * (1 - self.beta_vec)

            hidden_bar = self.cal_hidden_bar(hidden, L)

            R = torch.matmul(hidden_bar, self.trans_r_1) # B x S, S x R -> B x R
            LR = torch.einsum('br,br->br', L, R) # B x R, B x R -> B x R

            hidden_language = torch.matmul(LR, self.trans_r_2.transpose(0, 1)) # B x R, R x S  -> B x S
            hidden_wildcard = torch.matmul(hidden_bar, self.trans_wildcard) # B x S, S x S -> B x S
            hidden_ = hidden_language + hidden_wildcard

            if self.clip_neg:  # B x S -> B x S
                hidden_ = nn.functional.relu(hidden_)

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
        if self.farnn == 1:
            self.batch_h1  = self.h1.unsqueeze(0).repeat(B, 1)  # B x S

        hidden = self.h0.unsqueeze(0).repeat(B, 1) # B x S

        all_hidden = torch.zeros((B, L, self.S + self.AS)).cuda() if self.is_cuda else torch.zeros((B, L, self.S + self.AS))

        for i in range(L):
            emb = all_emb[:, i, :]  # B x D

            L_generalized = torch.matmul(emb, self.embed_r_generalized)  # B x D, D x R -> B x R

            if self.additional_nonlinear == 'relu':
                L_generalized = nn.functional.relu(L_generalized)
            elif self.additional_nonlinear == 'tanh':
                L_generalized = torch.tanh(L_generalized)
            elif self.additional_nonlinear == 'sigmoid':
                L_generalized = torch.sigmoid(L_generalized)
            else:
                pass

            L_regex = all_regex_emb[:, i, :]
            L = L_regex * self.beta_vec + L_generalized * (1 - self.beta_vec)

            hidden_bar = self.cal_hidden_bar(hidden, L)

            Tr1 = torch.einsum('br,rs->brs', L, self.trans_r_1.transpose(1, 0))  # B x R, R x S -> B x R x S
            Tr = torch.einsum('sr,brj->bjs', self.trans_r_2, Tr1)  # S x R, B x R x S -> B x S x S
            Tr = Tr + self.trans_wildcard # B x S x S + S x S  -> B x S x S
            hidden_ = self.maxmul(hidden_bar, Tr)  # B x S,  B x S x S  -> B x S

            if self.clip_neg:  # B x S -> B x S
                hidden_ = nn.functional.relu(hidden_)

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


class IntentIntegrateSaperateBidirection_B(nn.Module):
    def __init__(self, pretrained_embed=None, forward_params=None, backward_params=None, config=None, h1_forward=None, h1_backward=None):
        '''
        :param pretrained_embed:
        :param forward_params:  Dict
        :param backward_params:
        :param h1_forward:
        :param h1_backward:
        '''
        super(IntentIntegrateSaperateBidirection_B, self).__init__()

        print('D1 forward', forward_params['D1'].sum())
        print('D2 forward', forward_params['D2'].sum())
        print('V forward', forward_params['V_embed_extend'].sum())
        print('D1 backward', backward_params['D1'].sum())
        print('D2 backward', backward_params['D2'].sum())
        print('V backward', backward_params['V_embed_extend'].sum())

        self.fsa_rnn_forward = FSARNNIntegrateEmptyStateSaperateGRU(pretrained_embed=forward_params['pretrain_embed_extend'],
                                                                    trans_r_1=forward_params['D1'],
                                                                    trans_r_2=forward_params['D2'],
                                                                    embed_r=forward_params['V_embed_extend'],
                                                                    trans_wildcard=forward_params['wildcard_mat'],
                                                                    config=config,
                                                                    h1=h1_forward,)

        self.fsa_rnn_backward = FSARNNIntegrateEmptyStateSaperateGRU(pretrained_embed=backward_params['pretrain_embed_extend'],
                                                                     trans_r_1=backward_params['D1'],
                                                                     trans_r_2=backward_params['D2'],
                                                                     embed_r=backward_params['V_embed_extend'],
                                                                     trans_wildcard=backward_params['wildcard_mat'],
                                                                     config=config,
                                                                     h1=h1_backward,)

        self.S, self.R = forward_params['D1'].shape
        self.V, self.D = pretrained_embed.shape
        self.clamp_score = bool(config.clamp_score)
        self.clamp_hidden = bool(config.clamp_hidden)
        self.wfa_type = config.wfa_type

        if not config.random:

            forward_mat = torch.from_numpy(forward_params['mat']).float()
            forward_mat = torch.cat([forward_mat, torch.randn(config.additional_state, forward_mat.size()[1]) * config.random_noise])
            backward_mat = torch.from_numpy(backward_params['mat']).float()
            backward_mat = torch.cat([backward_mat, torch.randn(config.additional_state, backward_mat.size()[1]) * config.random_noise])

            self.mat_forward = nn.Parameter(forward_mat, requires_grad=bool(config.train_linear))
            self.bias_forward = nn.Parameter(torch.from_numpy(forward_params['bias']).float(), requires_grad=bool(config.train_linear))
            self.mat_backward = nn.Parameter(backward_mat, requires_grad=bool(config.train_linear))
            self.bias_backward = nn.Parameter(torch.from_numpy(backward_params['bias']).float(), requires_grad=bool(config.train_linear))
        else:
            self.mat_forward = nn.Parameter(torch.randn((forward_params['mat'].shape[0] + config.additional_state, forward_params['mat'].shape[1]) ).float(), requires_grad=bool(config.train_linear))
            self.bias_forward = nn.Parameter(torch.randn(forward_params['bias'].shape).float(), requires_grad=bool(config.train_linear))
            self.mat_backward = nn.Parameter(torch.randn((backward_params['mat'].shape[0] + config.additional_state, backward_params['mat'].shape[1]) ).float(), requires_grad=bool(config.train_linear))
            self.bias_backward = nn.Parameter(torch.randn(backward_params['bias'].shape).float(), requires_grad=bool(config.train_linear))

    def forward(self, input_forward, input_backward, lengths):
        if self.wfa_type == 'forward':
            out_forward = self.fsa_rnn_forward.forward(input_forward, lengths)
            out_backward = self.fsa_rnn_backward.forward(input_backward, lengths)
        elif self.wfa_type == 'viterbi':
            out_forward = self.fsa_rnn_forward.viterbi(input_forward, lengths)
            out_backward = self.fsa_rnn_backward.viterbi(input_backward, lengths)
        else:
            raise NotImplementedError()

        B, L = input_forward.size()
        last_hidden_backward = out_backward[torch.arange(B), lengths - 1, :]  # select all last hidden
        last_hidden_forward = out_forward[torch.arange(B), lengths - 1, :]  # select all last hidden

        if self.clamp_hidden:
            last_hidden_backward = torch.clamp(last_hidden_backward, 0, 1)
            last_hidden_forward = torch.clamp(last_hidden_forward, 0, 1)

        scores_backward = torch.matmul(last_hidden_backward, self.mat_backward) + self.bias_backward
        scores_forward = torch.matmul(last_hidden_forward, self.mat_forward) + self.bias_forward
        scores = (scores_backward + scores_forward)/2 # average

        if self.clamp_score:
            scores = torch.clamp(scores, 0, 1)

        return scores


class IntentIntegrateSaperate_B(nn.Module):
    def __init__(self, pretrained_embed=None, trans_r_1=None, trans_r_2=None, embed_r=None, trans_wildcard=None, config=None,
                mat=None, bias=None, h1_forward=None,):

        super(IntentIntegrateSaperate_B, self).__init__()

        self.fsa_rnn = FSARNNIntegrateEmptyStateSaperateGRU(pretrained_embed=pretrained_embed,
                                                             trans_r_1=trans_r_1,
                                                             trans_r_2=trans_r_2,
                                                             embed_r=embed_r,
                                                             trans_wildcard=trans_wildcard,
                                                             config=config,
                                                             h1=h1_forward,)

        self.S, self.R = trans_r_1.shape
        self.V, self.D = pretrained_embed.shape
        self.clamp_score = bool(config.clamp_score)
        self.clamp_hidden = bool(config.clamp_hidden)
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

        if self.clamp_hidden:
            last_hidden = torch.clamp(last_hidden, 0, 1)

        scores = torch.matmul(last_hidden, self.mat) + self.bias

        if self.clamp_score:
            scores = torch.clamp(scores, 0, 1)

        return scores