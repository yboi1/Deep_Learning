import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import string

training_data = [("The dog ate the apple".split(),["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
                 ("I hate you".split(),["NN","V","NN"]),
                 ("I love children".split(),["NN","V","NN"])]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 词嵌入
word_to_idx = {}
tag_to_idx = {}
idx_to_tag  = {}

for context, tag in training_data:
    for word in context:
        if word not in word_to_idx:
            word_to_idx[word.lower()] = len(word_to_idx)
    for label in tag:
        if label not in tag_to_idx:
            tag_to_idx[label.lower()] = len(tag_to_idx)
            # idx_to_tag[len(tag_to_idx) - 1

alphabet = 'abcdefghijklmnopqrstuvwxyz'
char_to_idx = {}
for i in range(len(alphabet)):
    char_to_idx[alphabet[i]] = i

def make_seq(vocab,dicx):
    # 将单词序列转化为数字序列
    id_indx = [dicx[i.lower()] for i in vocab]
    id_indx = torch.LongTensor(id_indx)

    return id_indx



class CharLSTM(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        self.char_embedding = nn.Embedding(n_char, char_dim)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, batch_first=True)

    def forward(self, x):
        x = self.char_embedding(x)
        _, h = self.char_lstm(x)
        return h[0]

class LSTMTagger(nn.Module):
    def __init__(self, n_word, n_char, char_dim, n_dim, char_hidden,
                 n_hidden, n_tag):
        super(LSTMTagger, self).__init__()
        self.word_embedding = nn.Embedding(n_word, n_dim)
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
        self.lstm = nn.LSTM(n_dim+char_hidden, n_hidden, batch_first=True)
        self.linear1 = nn.Linear(n_hidden, n_tag)

    # def forward(self,x, word_data):
    #     word = [i for i in word_data]
    #     char = torch.FloatTensor()
    #     for each in word:
    #         word_list = []
    #         for letter in each:
    #             word_list.append(char_to_idx[letter.lower()])
    #         word_list = torch.LongTensor(word_list)
    #         word_list=  word_list.unsqueeze(0)
    #         tempchar = self.char_lstm(word_list.to(device))
    #         char = torch.cat((char, tempchar.cpu().data), 0)
    #
    #     char = char.squeeze(1)
    #     char = char.to(device)
    #
    #     x = self.word_embedding(x)
    #     x = torch.cat((x, char), 1)
    #     # 扩大维度： 输入要带上batch_size
    #     x = x.unsqueeze(0)
    #     x, _ = self.lstm(x)
    #     x = x.squeeze(0)
    #     x = self.linear1(x)
    #     y = F.log_softmax(x)
    #     return y

    def forward(self, x, word):
        char = []
        for w in word:
            char_list = make_seq(w, char_to_idx)
            char_list = char_list.unsqueeze(1)
            char_infor = self.char_lstm(Variable(char_list))
            char.append(char_infor)
        char = torch.stack(char, dim=0)
        x = self.word_embed(x)  # batch,seq,word_dim
        x = x.permute(1, 0, 2)
        x = torch.cat((x, char), dim=2)
        x, _ = self.word_lstm(x)  # seq,batch,word_hidden

        s, b, h = x.shape
        x = x.view(-1, h)
        out = self.classify(x)

        return out

model = LSTMTagger(len(word_to_idx), len(char_to_idx), 10, 100, 50, 128, len(tag_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

for e in range(300):
    train_loss = 0
    for word, tag in training_data:
        word_list = make_seq(word, word_to_idx).unsqueeze(0)
        tag = make_seq(tag, tag_to_idx)

        # forward
        out = model(word_list, word)
        #out = model(word_list, word)
        loss = criterion(out, tag)
        train_loss += loss.item()

        #backWard
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (e + 1) % 50 == 0:
        print(f'Epoch: {e+1}, Loss: {train_loss/len(training_data)}')


# test
model.eval()
test_sent = 'I love you'
test = make_seq(test_sent.split(), word_to_idx).unsqueeze(0)
out = model(Variable(test), test_sent.split())
_,pred = torch.max(out,1)
print([idx_to_tag[int(i)] for i in pred])

