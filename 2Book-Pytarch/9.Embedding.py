import torch
import torch.nn.functional as F
from torch import nn
# 词嵌入
word_to_ix = {'hello':0, 'world':1}
embeds = nn.Embedding(2, 5) # 2 * 5 ： 词个数， 每个词的维度
hello_idx = torch.LongTensor([word_to_ix['hello']])
# hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)


# N Garm 模型
# 由前面几个词来预测下一个词的概率
# 马尔科夫假设： 一个词只与前面的几个词有关系

# 单词预测实现
CONTEXT_SIZE = 2    # 通过前面几个词来预测
EMBEDDING_DIM = 10  # 词嵌入的维数
test_sentence = '''
When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a tattered weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserved thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.'''.split()

# 将数据分为三组
trigram = [((test_sentence[i], test_sentence[i+1]), test_sentence[i+2])
           for i in range(len(test_sentence)-2)]

# 将每个单词编码
vocb = set(test_sentence)   #通过set去重
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

class NgramModel(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        super(NgramModel, self).__init__()

        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size*n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob


model = NgramModel(len(vocb), CONTEXT_SIZE, EMBEDDING_DIM)
criterion = nn.NLLLoss()  # 负对数似然损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for context, target in trigram:
    context_idxs = torch.tensor([word_to_idx[w] for w in context], dtype=torch.long)
    model.zero_grad()

    log_probs = model(context_idxs)
    loss = criterion(log_probs, torch.tensor([word_to_idx[target]], dtype=torch.long))
    loss.backward()
    optimizer.step()

print('Training completed!')


word, label = trigram[3]
word = torch.LongTensor([word_to_idx[i] for i in word])
out = NgramModel(word, CONTEXT_SIZE, EMBEDDING_DIM)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.data[0][0]]
print('read word is {}, predict word is {}'.format(label, predict_word))


