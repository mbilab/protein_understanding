import torch
from torch import nn

from bert.train import IGNORE_INDEX

class Seq2Seq(nn.Module):

    def __init__(self, model, device):
        super(Seq2Seq, self).__init__()

        self.model = model
        self.device = device
        self.Bidirectional_1 = nn.LSTM(128, 200, 2, bidirectional=True)
        self.dropout_1 = nn.Dropout()
        self.Bidirectional_2 = nn.LSTM(400, 200, 2, bidirectional=True)
        self.dropout_2 = nn.Dropout()
        self.Bidirectional_3 = nn.LSTM(400, 200, 2, bidirectional=True)
        self.dropout_3 = nn.Dropout()
        self.time_distributed = nn.Linear(400 + 128, 64)
        # @!!!!!!!!!!!!!!!!!!!!!@
        # The source code attached to the paper is 3, but the dataset is 2
        self.ppi_output = nn.Linear(64, 2)
        self.act = nn.Softmax(dim=2)
        #self.loss_function = nn.MSELoss()
        self.loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, inputs, targets):

        outputs = self.model(inputs)

        # the front two ouputs is going to be ignored
        # encoded_sources: (batch_size, seq_len, embed_size)
        mlm_outputs, nsp_outputs, encoded_sources = outputs
        x = encoded_sources
        input_b = encoded_sources.clone().detach().requires_grad_(True)
        assert x.shape[2] == 128
        assert x.shape[1] == 600 == targets.shape[1] # During the test, the length is fixed to 600

        # Transpose, so, x: (seq_len, batch_size, embed_size)
        x = encoded_sources.transpose(0, 1)
        x, _ = self.Bidirectional_1(x)
        x = self.dropout_1(x)
        x, _ = self.Bidirectional_2(x)
        x = self.dropout_2(x)
        x, _ = self.Bidirectional_3(x)
        x = self.dropout_3(x)

        # input_a: (batch_size, seq_len, embed_size)
        input_a = x.transpose(0, 1)
        input_cat = torch.cat([input_a, input_b], dim=2)
        out = self.time_distributed(input_cat)
        out = self.ppi_output(out)
        out = self.act(out)
        _out = out.reshape(-1,2)
        _targets = targets.reshape(-1)
        print(out)
        '''
        out_ = out.clone().reshape(-1,2)
        targets_ = targets.clone().reshape(-1,2)
        for  t in targets_:
            if 0. == t[0] and 0. == t[1]:
                t[1] = 2
        targets_ = torch.narrow(targets_, 1, 1, 1).squeeze().long()
        '''
        loss = self.loss_function(_out, _targets)
        predictions = out

        return predictions, loss.unsqueeze(dim=0)
