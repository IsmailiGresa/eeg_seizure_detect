from models.feature_extractor.spectrogram_feature import *
from models.transformer.self_attention import MultiHeadAttention

class CNN2D_BiGRU(nn.Module):
    def __init__(self, args, device):
        super(CNN2D_BiGRU, self).__init__()
        self.args = args
        self.device = device

        self.num_layers = 1
        self.hidden_dim = 64
        self.attention_dim = args.attention_dim
        self.num_heads = args.multi_head_num
        self.dropout = args.dropout
        self.num_data_channel = args.num_channel
        self.feature_extractor = args.enc_model

        if self.feature_extractor == "raw":
            self.feat_model = None

        self.activation = nn.ReLU(inplace=True)

        def conv2d_bn(inp, oup, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(oup),
                self.activation,
                nn.Dropout(self.dropout),
            )

        if args.enc_model == "raw":
            self.features = nn.Sequential(
                conv2d_bn(1, 64, (1, 51), (1, 4), (0, 25)),
                conv2d_bn(64, 128, (1, 21), (1, 2), (0, 10)),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                conv2d_bn(128, 256, (1, 9), (1, 2), (0, 4)),
            )

        self.agvpool = nn.AdaptiveAvgPool2d((1, 8))

        self.bigru = nn.GRU(
            input_size=256,
            hidden_size=self.hidden_dim // 2,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        self.self_attention = MultiHeadAttention(dim=self.hidden_dim, num_heads=self.num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Linear(in_features=64, out_features=args.output_dim, bias=True),
        )


    def forward(self, x, targets=None, seq_lengths=None, target_lengths=None):
        x = x.squeeze(1).permute(0, 2, 1).contiguous()

        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.agvpool(x)
        x = torch.squeeze(x, 2)
        x = x.permute(0, 2, 1)

        output, _ = self.bigru(x)

        output, _ = self.self_attention(output, output, output)

        output = output[:, -1, :]
        output = self.classifier(output)

        return output

    def init_state(self, device, batch_size=32):
        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(device)
        return h_0
