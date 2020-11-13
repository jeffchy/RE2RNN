import torch
import torch.nn as nn


class FSAIntegrateOnehot(nn.Module):
    def __init__(self, fsa_tensor=None, is_cuda=True):
        """
        Parameters
        ----------
        pretrained_embed: pretrained glove embedding,  V x D, numpy array
        trans_r_1: Tensor decomposition components 1, S x R (state x rank) numpy array
        trans_r_2: Tensor decomposition components 2, S x R (state x rank) numpy array
        embed_r: Tensor decomposition components 0, V x R (vocab size x R) numpy array
        config: config
        """
        super(FSAIntegrateOnehot, self).__init__()

        # self.is_cuda = torch.cuda.is_available()
        self.is_cuda = torch.cuda.is_available() if is_cuda else is_cuda

        V, S, S = fsa_tensor.shape
        self.S = S
        self.h0 = self.hidden_init() # S hidden state dim should be equal to the state dim
        self.fsa_tensor = nn.Parameter(torch.from_numpy(fsa_tensor).float(), requires_grad=True) # V x S x S

    def forward(self, input, lengths):
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

        B, L = input.size() # B x L
        hidden = self.h0.unsqueeze(0).repeat(B, 1) # B x S
        all_hidden = torch.zeros((B, L, self.S)).cuda() if self.is_cuda else torch.zeros((B, L, self.S))

        for i in range(L):
            inp = input[:, i] # B
            Tr = self.fsa_tensor[inp] # B x S x S
            hidden = torch.einsum('bs,bsj->bj', hidden, Tr) # B x R, B x R -> B x R
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
        B, L = input.size() # B x L
        hidden = self.h0.unsqueeze(0).repeat(B, 1) # B x S
        all_hidden = torch.zeros((B, L, self.S)).cuda() if self.is_cuda else torch.zeros((B, L, self.S))

        for i in range(L):
            inp = input[:, i] # B
            Tr = self.fsa_tensor[inp] # B x S x S
            hidden = self.maxmul(hidden, Tr)  # B x S,  B x S x S  -> B x S
            all_hidden[:, i, :] = hidden

        return all_hidden

    def hidden_init(self):
        hidden = torch.zeros((self.S), dtype=torch.float)
        hidden[0] = 1.0
        hidden = hidden.cuda() if self.is_cuda else hidden
        return hidden


class IntentIntegrateOnehot(nn.Module):
    def __init__(self, fsa_tensor, config=None,
                mat=None, bias=None, is_cuda=True):
        super(IntentIntegrateOnehot, self).__init__()

        self.fsa_rnn =  FSAIntegrateOnehot(fsa_tensor, is_cuda)
        self.mat = nn.Parameter(torch.from_numpy(mat).float(), requires_grad=bool(config.train_linear))
        self.bias = nn.Parameter(torch.from_numpy(bias).float(), requires_grad=bool(config.train_linear))
        self.clamp_score = bool(config.clamp_score)
        self.clamp_hidden = bool(config.clamp_hidden)
        self.wfa_type = config.wfa_type

    def forward(self, input, lengths):
        if self.wfa_type == 'viterbi':
            out = self.fsa_rnn.viterbi(input, lengths)
        else:
            out = self.fsa_rnn.forward(input, lengths)

        B, L = input.size()
        last_hidden = out[torch.arange(B), lengths - 1, :]  # select all last hidden
        if self.clamp_hidden:
            last_hidden = torch.clamp(last_hidden, 0, 1)
        scores = torch.matmul(last_hidden, self.mat) + self.bias
        if self.clamp_score:
            scores = torch.clamp(scores, 0, 1)
        return scores