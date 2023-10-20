import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
# from tensorboardX import SummaryWriter

EPOCH = 150
BATCH_SIZE = 32
TIME_STEP = 512
INPUT_SIZE = 12
LR = 0.0005

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
        )

        self.out = nn.Linear(512,18)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:,-1,:])
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# read pickle file
# training data

data_dir = 'H20_train_eval_split/'
test_sub = 'H_005'
train_feat_file = data_dir + test_sub + '_train_features.pickle'
train_label_file = data_dir + test_sub + '_train_labels.pickle'
test_feat_file = data_dir + test_sub + '_test_features.pickle'
test_label_file = data_dir + test_sub + '_test_labels.pickle'
eval_feat_file = data_dir + test_sub + '_eval_features.pickle'
eval_label_file = data_dir + test_sub + '_eval_labels.pickle'

with open(train_feat_file,'rb') as handle1:
    X_train = pickle.load(handle1)
with open(train_label_file,'rb') as handle2:
    y_train = pickle.load(handle2)
with open(test_feat_file,'rb') as handle3:
    X_test = pickle.load(handle3)
with open(test_label_file,'rb') as handle4:
    y_test = pickle.load(handle4)
with open(eval_feat_file,'rb') as handle5:
    X_eval = pickle.load(handle5)
with open(eval_label_file,'rb') as handle6:
    y_eval = pickle.load(handle6)

y_train = numpy.array(y_train)
y_test = numpy.array(y_test)
y_eval = numpy.array(y_eval)

print('Train:',X_train.shape,y_train.shape,'test:',X_test.shape,y_test.shape,'eval:',X_eval.shape,y_eval.shape)

X_train = torch.Tensor(X_train).to(device)
y_train = torch.Tensor(y_train).to(device)
X_test = torch.Tensor(X_test).to(device)
X_eval = torch.Tensor(X_eval).to(device)
y_eval = torch.Tensor(y_eval).to(device)
train_data = torch.utils.data.TensorDataset(X_train,y_train)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
eval_data = torch.utils.data.TensorDataset(X_eval,y_eval)
eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, shuffle=True)
rnn = RNN().to(device)
optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

def evaluate(model):
    model.eval()
    eval_sum = 0
    eval_num = 0
    for step, (eval_x,eval_y) in enumerate(eval_loader):
        eval_output = model(eval_x)
        eval_pred_y = torch.max(eval_output, 1)[1].data.cpu().numpy()
        eval_y = eval_y.cpu().numpy()
        batch_acc_sum = float((eval_pred_y == eval_y).astype(int).sum())
        batch_num = eval_y.size
        eval_sum += batch_acc_sum
        eval_num += batch_num
    return eval_sum / eval_num

best_eval_acc = 0
for epoch in range(EPOCH):
    print('EPOCH:',epoch)
    for step, (train_x,train_y) in enumerate(train_loader):
        rnn.train()
        # print(train_x.shape,train_y.shape)
        output = rnn(train_x)
        loss = loss_func(output, train_y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%20==0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())
    if epoch>50 and epoch%3==0:
        eval_acc = evaluate(rnn)
        print('eval accuracy: %.2f' % eval_acc)
        if eval_acc>best_eval_acc:
            best_model = rnn
            best_eval_acc = eval_acc

best_model.eval()
test_output = rnn(X_test)
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
accuracy = float((pred_y == y_test).astype(int).sum()) / float(y_test.size)
print('| test accuracy: %.2f' % accuracy)
res = confusion_matrix(y_test,pred_y,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
print(res)

