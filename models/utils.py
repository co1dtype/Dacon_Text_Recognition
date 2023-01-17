import torch.nn as nn
from torchvision.models import resnet152

class RecognitionModel(nn.Module):
    def __init__(self, num_chars, rnn_hidden_size=256):
        super(RecognitionModel, self).__init__()
        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size

        # CNN Backbone
        resnet = resnet152(pretrained=True)
        # CNN Feature Extract
        resnet_modules = list(resnet.children())[:-3]
        self.feature_extract = nn.Sequential(
            *resnet_modules,
            nn.Conv2d(1024, 1024, kernel_size=(3, 6), stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.linear1 = nn.Linear(8192, rnn_hidden_size)

        # RNN
        self.rnn = nn.RNN(input_size=rnn_hidden_size,
                          hidden_size=rnn_hidden_size,
                          bidirectional=True,
                          batch_first=True)
        self.linear2 = nn.Linear(self.rnn_hidden_size * 2, num_chars)

    def forward(self, x):
        # CNN
        x = self.feature_extract(x)  # [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]

        batch_size = x.size(0)
        T = x.size(1)
        x = x.view(batch_size, T, -1)  # [batch_size, T==width, num_features==channels*height]
        x = self.linear1(x)

        # RNN
        x, hidden = self.rnn(x)

        output = self.linear2(x)
        output = output.permute(1, 0, 2)  # [T==10, batch_size, num_classes==num_features]

        return output

def get_model(model_str: str):
    if model_str == 'RCNN':
        return RecognitionModel

    raise Exception(f"'{model_str}': This Model does not exist.")
