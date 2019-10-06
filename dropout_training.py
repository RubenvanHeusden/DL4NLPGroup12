from models.dropoutlstm import DropoutLSTM
import tqdm
import torch
import numpy as np
# import torch.nn.functional as F
from torch import nn
import torch.optim as optim
# import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from batchwrapper import BatchWrapper
from models.simplelstm import SimpleLSTM
from models.dropoutlstm import DropoutLSTM
from models.bilstm import BiLSTM
from models.kimcnn import KimCNN
from models.simplecnn import SimpleCNN
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from nltk.tokenize import  word_tokenize

torch.manual_seed(42)

num_classes = 90
batch_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 50

# We read in our stop words file into a list
def read_stop_words(f_path):
    with open(f_path, 'r') as f:
        return f.read().split("\n")

stopwords = read_stop_words("data/stopwords")
# Specifies what we want to do with our document column in the
# csv. Here we want to build a vocabulary from the text, remove stop words
# and specify a custom tokenizer (the nltk one in this case)
text = Field(sequential=True, tokenize=word_tokenize,
             stop_words=stopwords,
             lower=True)

# Here we specify the labels we are going to use as targets, in our case these
# are n-hot vectors of length 90
labels = Field(sequential=False, use_vocab=False,is_target=True)

# specify the types for all 90 classes
classes = [("class_%d"%x, labels) for x in range(num_classes)]
class_names = list(map(lambda x :x[0], classes))

# create the list with datatypes for all columns in one row of data
datafields = [("id", None),
                   ("document", text),*classes]



#Here we actually read in the data from the data folder and we split
# it into three sets
trn, val, tst = TabularDataset.splits(
               path="data",
               train='train.csv', validation='valid.csv',test="test.csv",
               format='csv',
               skip_header=False,
               fields=datafields,
                csv_reader_params = {'delimiter':"\t", "quotechar":"|"})


# Build the vocabulary using vectors from glove, the vectors argument
# supports many different sets of embeddings and you can also set it to
# load from a local file if you already have the embeddings downloaded
text.build_vocab(trn,vectors="glove.6B.300d")


# make an iterator to loop over the data in batches, also applies padding
train_iter, val_iter = BucketIterator.splits(
 (trn, val),
 batch_sizes=(batch_size, batch_size),
 device=device,
 sort_key=lambda x: len(x.document),
 sort_within_batch=True,
 repeat=False
)

# Make a seperate iterator for the test data in which we dont sort
# depending on length but just keep it as it is
test_iter = Iterator(tst, batch_size=batch_size,
                     device=device,
                     sort=False,
                     sort_within_batch=False,
                     repeat=False)


# Use the BatchWrapper to make the output of the iterator more intuitive
train_dl = BatchWrapper(train_iter, "document", class_names)
valid_dl = BatchWrapper(val_iter, "document", class_names)
test_dl = BatchWrapper(test_iter, "document", class_names)


# Specific model that we use for the training procedure
lstm_model = DropoutLSTM(hidden_dim=256, vocab_size=len(text.vocab),
                        embedding_dim=300, weight_matrix=text.vocab.vectors,
                    dvc=device, tr_embed=False)
lstm_model = lstm_model.to(device)

# max_length = max([len(trn[x].document) for x in range(len(trn))])
# cnn = KimCNN(embedding_dim=300, vocab_size=len(text.vocab),
#                 max_size=max_length, weight_matrix=text.vocab.vectors)

# from torch.distributions.bernoulli import  Bernoulli
# bernoulli = Bernoulli(0.2)
# mask = bernoulli.sample((1024, 1))
# for name, param in lstm_model.lstm.named_parameters():
#     if "weight_ih" in name:
#         getattr(lstm_model.lstm, name).data = param.data*(mask.repeat(1, 300))
#         print(getattr(lstm_model.lstm, name).data)

def train(train_data, val_data, test_data,
          model=None,
          num_epochs=5, lr=1e-2, logging=False):

    dropout = 0.4
    p = 1 - dropout
    n = len(trn)
    w1 = None
    w2 = None
    b = None


    for name, param in model.lstm.named_parameters():
        if "weight_ih" in name:
            w1 = param
    for name, param in model.output_layer.named_parameters():
        if "weight" in name:
            w2 = param
        elif "bias" in name:
            b = param

    # maybe make this model lstm
    opt = optim.Adam(model.parameters(), lr=0.001,
                     weight_decay=1e-5)  # 1e-2  +- batch_size/64*0.01
    loss_func = nn.BCEWithLogitsLoss()  # Experimental: weigh loss with class occurrence

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        model.train()
        print("Running Epoch {}".format(epoch))
        for x, y in tqdm.tqdm(train_data):
            opt.zero_grad()
            preds = model(x)
            loss = loss_func(preds, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(trn)

        val_loss = 0.0
        model.eval()
        for x, y in val_data:
            preds = model(x)
            loss = loss_func(preds, y).mean()
            val_loss += loss.item() * x.size(0)

        val_loss /= len(val)
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
        print()


    test_preds = []
    targets = []

    one_class_preds = []
    one_class_targets = []

    multi_class_preds = []
    multi_class_targets = []

    model.eval()
    sigmoid = nn.Sigmoid()
    for x, y in tqdm.tqdm(test_data):
        #preds = model(x[:max_length, :])
        preds = model(x)
        preds = sigmoid(preds)>0.5
        preds = preds.cpu().data.numpy()
        test_preds.append(preds)
        target = y.cpu().data.numpy()
        targets.append(target)

        multi_cls_mask = target.sum(1) > 1
        single_cls_mask = 1-multi_cls_mask

        one_class_preds.append(preds[single_cls_mask])
        one_class_targets.append(target[single_cls_mask])

        multi_class_preds.append(preds[multi_cls_mask])
        multi_class_targets.append(target[multi_cls_mask])

    targets = np.concatenate(targets, axis=0)
    preds = np.concatenate(test_preds, axis=0)

    o_targets = np.concatenate(one_class_targets, axis=0)
    o_preds = np.concatenate(one_class_preds, axis=0)

    m_targets = np.concatenate(multi_class_targets, axis=0)
    m_preds = np.concatenate(multi_class_preds, axis=0)
    print(o_preds.sum(), o_preds.shape)
    print("Micro Precision score %f"%precision_score(targets, preds , average="micro"))
    print("Micro Recall score %f"%recall_score(targets, preds , average="micro"))
    print("Micro F1 score %f"%f1_score(targets, preds , average="micro"))
    print("Accuracy score %f"%accuracy_score(targets, preds))

    print("Micro Precision score single class %f"%precision_score(o_targets, o_preds, average="micro"))
    print("Micro Recall score single class %f"%recall_score(o_targets, o_preds , average="micro"))
    print("Micro F1 score single class %f"%f1_score(o_targets, o_preds, average="micro"))
    print("Accuracy score single class %f"%accuracy_score(o_targets, o_preds))

    print("Micro Precision score multi class %f"%precision_score(m_targets, m_preds, average="micro"))
    print("Micro Recall score multi class %f"%recall_score(m_targets, m_preds , average="micro"))
    print("Micro F1 score multi class %f"%f1_score(m_targets, m_preds, average="micro"))
    print("Accuracy score multi class %f"%accuracy_score(m_targets, m_preds))

train(train_data=train_dl, val_data=valid_dl,
      test_data=test_dl,
      model=lstm_model,num_epochs=num_epochs)
