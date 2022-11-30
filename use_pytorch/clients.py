import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from .getData import GetDataSet
import transformer as tf
import math
import time
import pickle
from sklearn.utils import shuffle
#from utils import gen_aligned_tensor, next_batch
from sklearn.cluster import KMeans
#np.random.seed(32)

class client(object):
    def __init__(self, trainDataSet, length,dev):
        self.train_ds = trainDataSet
        self.length=length
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters,PAD_IDX,dev):
        dataset_len = len(self.train_ds)
        # train_proportion = 0.6
        # val_proportion = 0.2
        train_seq=self.train_ds
        # train_seq = self.train_ds[:int(dataset_len * train_proportion)]
        # val_seq = self.train_ds[
        #           int(dataset_len * train_proportion):int(dataset_len * (train_proportion + val_proportion))]
        # test_seq = self.train_ds[int(dataset_len * (train_proportion + val_proportion)):]
        Net.load_state_dict(global_parameters, strict=True)
        trajectoryVecs = []
        min_loss, worse_round = 1e8, 0
        #self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            total_start_time = time.time()
            train_losses, val_losses = 0, 0

            for batch in next_batch(shuffle(train_seq), localBatchSize):
                encoder_inputs, valid_len = gen_aligned_tensor(batch, fill_value=PAD_IDX)
                encoder_inputs = torch.from_numpy(encoder_inputs).float().to(dev)  # (bs, max_valid_len, num_feat)
                valid_len = torch.from_numpy(valid_len).long().to(dev)  # (bs)
                embedding, train_loss, val_loss = tf.train(Net, opti, encoder_inputs, valid_len, PAD_IDX, dev)
                train_losses += train_loss
                val_losses += val_loss
            total_end_time = time.time()
            trajectoryVecs.append(embedding)
            print(f"Epoch: {epoch + 1}, Train loss: {train_losses:.3f}, Val loss: {val_losses:.3f}, "
                  F'Epoch Time: {(total_end_time - total_start_time):.3f}s')
            # for data, label in self.train_dl:
            #     data, label = data.to(self.dev), label.to(self.dev)
            #     preds = Net(data)
            #     loss = lossFun(preds, label)
            #     loss.backward()
            #     opti.step()
            #     opti.zero_grad()


        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, style, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.style=style

        self.test_data_loader = None
        if style==2:
            self.train_data, self.val_data, self.test_data = self.t_dataSetBalanceAllocation()
        elif style==1:
            self.train_data,self.val_data,self.test_data=self.s_dataSetBalanceAllocation()
        elif style==0:
            self.train_data, self.val_data, self.test_data = self.dataSetBalanceAllocation()
        else:
            self.train_data, self.val_data, self.test_data = self.st_dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):

        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid,self.style)

        #test_data = torch.tensor(mnistDataSet.test_data)
        #test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        #self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        #self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = shuffle(mnistDataSet.train_data)
        val_data=mnistDataSet.val_data
        test_data = mnistDataSet.test_data
        #train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients #// 2
        #shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)

        for i in range(self.num_of_clients):
            #shards_id1 = shards_id[i * 2]
            #shards_id2 = shards_id[i * 2 + 1]
            #data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            #data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            #label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            #label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            #local_data= np.vstack((data_shards1, data_shards2))
            #若正常划分，请运行如下代码：
            local_data=train_data[i*shard_size:(i+1)*shard_size]
            length=shard_size
            #若想要按照地域划分，请运行如下代码：
            # local_data=slice_data[i]
            # length= len(slice_data[i])

            #local_data=local_data.astype(float)
            #local_label = np.argmax(local_label, axis=1)
            # 若正常划分，请运行如下代码,并且将client的length属性取消掉：
            self.clients_set['client{}'.format(i)] = client(local_data,length,self.dev)
            # 若想要按照地域划分，请运行如下代码：
            #self.clients_set['client{}'.format(i)] = client(local_data, length,self.dev)
            # someone = client(TensorDataset(torch.tensor(local_data),self.dev))
            # self.clients_set['client{}'.format(i)] = someone
            # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            # local_label = np.argmax(local_label, axis=1)
            # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            # self.clients_set['client{}'.format(i)] = someone
        return train_data,val_data, test_data

    def s_dataSetBalanceAllocation(self):

        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid,self.style)

        #test_data = torch.tensor(mnistDataSet.test_data)
        #test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        #self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        #self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        val_data=mnistDataSet.val_data
        test_data = mnistDataSet.test_data
        #train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients #// 2
        #shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)


        lon_lat=[i[0][2:4] for i in train_data]#刚才改了
        clf = KMeans(n_clusters=self.num_of_clients)
        # 提取点集
        clf.fit(lon_lat)
        ret_n = clf.predict(lon_lat)
        slice_data = []
        for i in range(self.num_of_clients):
            slice_data.append([])
        for i in range(len(ret_n)):
            slice_data[ret_n[i]].append(train_data[i])
        # for i in np.unique(ret_n):
        #     slice_data.append([])
        #     for j in range(len(ret_n)):
        #         if ret_n[j]==i:
        #             slice_data[-1].append(train_data[j])

        #point_set = [(i[1:3] for i in i_[:] )for i_ in train_data]
       # s = clf.fit(lon_lat)  # 加载数据集合
        # numSamples = len(s)


        for i in range(self.num_of_clients):
            #shards_id1 = shards_id[i * 2]
            #shards_id2 = shards_id[i * 2 + 1]
            #data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            #data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            #label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            #label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            #local_data= np.vstack((data_shards1, data_shards2))
            #若正常划分，请运行如下代码：
            # local_data=train_data[i*shard_size:(i+1)*shard_size]
            # length=shard_size
            #若想要按照地域划分，请运行如下代码：
            local_data=slice_data[i]
            length= len(slice_data[i])

            #local_data=local_data.astype(float)
            #local_label = np.argmax(local_label, axis=1)
            # 若正常划分，请运行如下代码,并且将client的length属性取消掉：
            self.clients_set['client{}'.format(i)] = client(local_data, length,self.dev)
            # 若想要按照地域划分，请运行如下代码：
            #self.clients_set['client{}'.format(i)] = client(local_data, length,self.dev)
            # someone = client(TensorDataset(torch.tensor(local_data),self.dev))
            # self.clients_set['client{}'.format(i)] = someone
            # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            # local_label = np.argmax(local_label, axis=1)
            # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            # self.clients_set['client{}'.format(i)] = someone
        return train_data,val_data, test_data

    def t_dataSetBalanceAllocation(self):

        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid,self.style)

        # test_data = torch.tensor(mnistDataSet.test_data)
        # test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        # self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        # self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        val_data = mnistDataSet.val_data
        test_data = mnistDataSet.test_data
        # train_label = mnistDataSet.train_label

        tdate = pickle.load(open('../data/Tdate_split9', 'rb'))
        tdate=tdate[:len(train_data)]
        ret_n = [date for date in tdate]




        slice_data = []
        for i in range(7):
            slice_data.append([])
        for i in range(len(ret_n)):
            slice_data[(ret_n[i]+4)%7].append(train_data[i])
        # for i in np.unique(ret_n):
        #     slice_data.append([])
        #     for j in range(len(ret_n)):
        #         if ret_n[j]==i:
        #             slice_data[-1].append(train_data[j])

        # point_set = [(i[1:3] for i in i_[:] )for i_ in train_data]
        # s = clf.fit(lon_lat)  # 加载数据集合
        # numSamples = len(s)

        for i in range(6):
            # shards_id1 = shards_id[i * 2]
            # shards_id2 = shards_id[i * 2 + 1]
            # data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            # label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            # local_data= np.vstack((data_shards1, data_shards2))
            # 若正常划分，请运行如下代码：
            # local_data=train_data[i*shard_size:(i+1)*shard_size]
            # length=shard_size
            # 若想要按照地域划分，请运行如下代码：
            local_data = slice_data[i]
            length = len(slice_data[i])

            # local_data=local_data.astype(float)
            # local_label = np.argmax(local_label, axis=1)
            # 若正常划分，请运行如下代码,并且将client的length属性取消掉：
            self.clients_set['client{}'.format(i)] = client(local_data, length, self.dev)
            # 若想要按照地域划分，请运行如下代码：
            # self.clients_set['client{}'.format(i)] = client(local_data, length,self.dev)
            # someone = client(TensorDataset(torch.tensor(local_data),self.dev))
            # self.clients_set['client{}'.format(i)] = someone
            # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            # local_label = np.argmax(local_label, axis=1)
            # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            # self.clients_set['client{}'.format(i)] = someone
        return train_data, val_data, test_data

    def st_dataSetBalanceAllocation(self):

        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid,self.style)

        # test_data = torch.tensor(mnistDataSet.test_data)
        # test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        # self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        # self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        val_data = mnistDataSet.val_data
        test_data = mnistDataSet.test_data
        # train_label = mnistDataSet.train_label

        tdate = pickle.load(open('../data/Tdate_split2304', 'rb'))##敏感度分析需要修改这里
        tdate=tdate[:len(train_data)]
        ret_n = [date for date in tdate]




        slice_data = []
        for i in range(7):
            slice_data.append([])
        for i in range(len(ret_n)):
            slice_data[(ret_n[i]+4)%7].append(train_data[i])


        for i in range(6):
            lon_lat = [i[0][2:4] for i in slice_data[i]]
            clf = KMeans(n_clusters=5)
            # 提取点集
            clf.fit(lon_lat)
            ret_n = clf.predict(lon_lat)
            slice_data2 = []
            for i2 in range(self.num_of_clients):
                slice_data2.append([])
            for i2 in range(len(ret_n)):
                slice_data2[ret_n[i2]].append(train_data[i2])
            for i3 in range(5):
                local_data = slice_data2[i3]
                length = len(slice_data2[i3])
                self.clients_set['client{}'.format(i*5+i3)] = client(local_data, length, self.dev)


        return train_data, val_data, test_data

def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]

def gen_aligned_tensor(sequences, fill_value, max_len=None):
    """
    Generate an aligned tensor given a list of tuples with uneven length.

    :param sequences: a list containing the original sequences.
    :param fill_value: the value for filling the sequences.
    :param max_len: the maximal length of the given sequence.
    :return:
    """
    aligned_sequences = []
    valid_lens = []
    if not max_len:
        if (len(sequences)>1):
            max_len = max(*[len(s) for s in sequences])
        else:
            max_len = len(sequences[0])
    tuple_len = len(sequences[0][0])
    empty_tuple = tuple([fill_value] * tuple_len)
    for sequence in sequences:
        valid_len = len(sequence)
        aligned_sequences.append(sequence + [empty_tuple] * (max_len - valid_len))
        valid_lens.append(valid_len)
    return np.array(aligned_sequences), np.array(valid_lens)

if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


