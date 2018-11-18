import torch.nn as nn
import torch.nn.functional as F
import torch


class RelationNetwork(nn.Module):
    def __init__(self,  input_size, hidden_size):
        super(RelationNetwork, self).__init__()

        # self.cnn = cnn
        # self.wordembedding = wordembedding
        # for p in self.cnn.parameters():
        #     p.requires_grad = False
        # for p in self.wordembedding.parameters():
        #     p.requires_grad = False


        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)



    def forward(self, visual_emb, semantic_emb):

        # self.cnn.eval()
        # self.wordembedding.eval()

        # visual_emb, _ = self.cnn(image)
        # batch_size = image.shape[0]
        # semantic_emb = self.wordembedding(wd)
        # class_num = semantic_emb.shape[0]
        # visual_emb = visual_emb.unsqueeze(0).repeat(class_num, 1, 1)
        # semantic_emb = semantic_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        # semantic_emb = torch.transpose(visual_emb, 0, 1)
        dim = semantic_emb.shape[2]
        x = torch.cat((visual_emb, semantic_emb), 2).view(-1,dim * 2)


        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x


class AttributeNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.word_emb_transformer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, word_embeddings):
        semantic_emb = self.word_emb_transformer(word_embeddings)
        return semantic_emb

# class AttributeNetwork(nn.Module):
#
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.word_emb_transformer = nn.Sequential(
#             nn.Linear(input_size, output_size),
#             nn.ReLU(),
#             # nn.Linear(hidden_size, output_size),
#             # nn.ReLU()
#         )
#
#     def forward(self, word_embeddings):
#         semantic_emb = self.word_emb_transformer(word_embeddings)
#         return semantic_emb
#


