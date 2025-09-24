import torch

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(int(in_channels), num_labels, (1,1))

    def forward(self, embeddings):
        try:
            embeddings = embeddings.reshape(embeddings.shape[0], self.height, self.width, self.in_channels)
        except:
            raise ValueError(f"Error reshaping embeddings with shape {embeddings.shape}")
        embeddings = embeddings.permute(0,3,1,2)

        return self.classifier(embeddings)