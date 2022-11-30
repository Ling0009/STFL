import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
#from .Models import Mnist_2NN, Mnist_CNN
#from clients import ClientsGroup, client
from use_pytorch.clients import ClientsGroup,client
import transformer as tf
import pickle
from tasks import test_dest_pre
from tasks import CellClassifyDecoder
import math
import time
PAD_IDX = 0
ysum = 80#原本是50
xsum = 80#原本是50
num_loc = xsum*ysum


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=30, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.14, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=70, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=40, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='transformer', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=5, help='number of communications')
#这里为了训练快改了
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    training_epochs = 200
    early_stopping_round = 10

    # Network Parameters
    # 2 different sequences total
    batch_size = 40
    # the maximum steps for both sequences is 5
    max_n_steps = 186  # 2*bucket_size的整数倍
    # each element/frame of the sequence has dimension of 3
    frame_dim = 30

    PAD_IDX = 0

    # dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = "cpu"

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    elif args['model_name'] == 'transformer':
        net, optimizer = tf.build_model(frame_dim,num_loc, dev)
        #net = Mnist_CNN()

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    style=3#0代表原始方式，1代表空间方式，2代表时间方式,3代表时空联合

    myClients = ClientsGroup('transformer', args['IID'], args['num_of_clients'], style, dev)
    testDataLoader = myClients.test_data_loader
    train_seq=myClients.train_data
    #train_seq = train_seq[:len(train_seq)//10]
    val_seq=myClients.val_data
    #val_seq = val_seq[:len(val_seq) // 10]
    test_seq=myClients.test_data
    #test_seq = test_seq[:len(test_seq) // 10]

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
    np.random.seed(32)
    torch.manual_seed(32)
    torch.cuda.manual_seed(32)
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args['num_of_clients'])#每次都打乱顺序，随机挑选一个？
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        total_length = 0
        for client in tqdm(clients_in_comm):
            #sum_length = 0
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'],
                                                                         args['batchsize'],
                                                                         net,
                                                                       loss_func, opti, global_parameters,PAD_IDX,dev)
            #普通avg请删除下面两行：
            # sum_length = myClients.clients_set[client].length
            # total_length = total_length + sum_length
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    #若fvg原始版本，请运行如下一行代码：
                    sum_parameters[key] = var.clone()
                    #若根据数据集大小分配权重，运行下面两行代码：
                    #sum_parameters[key] = var.clone() * sum_length

                    #停停停我强烈怀疑下面这两行是错的不要随便换
                    #sum_parameters[key] = var.clone()*myClients.clients_set[client].length
                    # sum_length+=myClients.clients_set[client].length
            else:
                for var in sum_parameters:#其实这里的var应该是上面的key，太具有迷惑性了，打死他
                    # 若fvg原始版本，请运行如下一行代码：
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
                    # 若根据数据集大小分配权重，运行下面两行代码：
                    # sum_parameters[var] = sum_parameters[var] + local_parameters[var]*myClients.clients_set[client].length
                    # sum_length += myClients.clients_set[client].length


        for var in global_parameters:
            # 若fvg原始版本，请运行如下一行代码：
            global_parameters[var] = (sum_parameters[var] / num_in_comm)
            # 若根据数据集大小分配权重，运行下面两行代码：
            #global_parameters[var] = (sum_parameters[var] / (total_length))

        # with torch.no_grad():
        #     if (i + 1) % args['val_freq'] == 0:
        #         net.load_state_dict(global_parameters, strict=True)
        #         sum_accu = 0
        #         num = 0
        #         for data, label in testDataLoader:
        #             data, label = data.to(dev), label.to(dev)
        #             preds = net(data)
        #             preds = torch.argmax(preds, dim=1)
        #             sum_accu += (preds == label).float().mean()
        #             num += 1
        #         print('accuracy: {}'.format(sum_accu / num))
        #
        # if (i + 1) % args['save_freq'] == 0:
        #     torch.save(net, os.path.join(args['save_path'],
        #                                  '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
        #                                                                                         i, args['epoch'],
        #                                                                                         args['batchsize'],
        #                                                                                         args['learning_rate'],
        #                                                                                         args['num_of_clients'],
        #                                                                                         args['cfraction'])))
        #
# fout = open('./data/tmb2vec_data/traj_vec_normal_reverse', 'wb')
# pickle.dump(trajectoryVecs, fout)
model=net
model_type=1
#downstream parameters
output_embed_size = 128 if model_type == 7 \
        else model.encoder.size if model_type == 2 or model_type == 6 or model_type == 8\
        else model.emb_size
task_init_lr = 1e-3
task_batch_size = 40
task_num_epoch = 200
task_early_stopping_round = 10
finetune = True
device = dev

distance_method = 'euc'  # 'euc' or 'dot'
eta_time_divisor = 60 * 60 * 24

    # def _get_type():
    #     if model_type == 4 or model_type == 5:
    #         return str(model_type) + '_' + str(int(use_init))
    #     elif model_type == 3:
    #         return str(model_type) + '_' + str(int(use_init)) + str(int(use_syn))
    #     else:
    #         return str(model_type)

    # Evaluate on downstream tasks.
# if flag == 0:
#     decoder_model = FCDecoder(output_embed_size, num_loc)
# else:

decoder_model = CellClassifyDecoder(output_embed_size, num_loc, max_n_steps, is_transformer=model_type == 1)
score_series1, results1 = test_dest_pre(model, decoder_model, train_seq, val_seq, test_seq,
                                  num_epoch=task_num_epoch, init_lr=task_init_lr, device=device,
                                  batch_size=task_batch_size, finetune=finetune,
                                  early_stopping_round=task_early_stopping_round,
                                  frame_dim=frame_dim, max_n_steps=max_n_steps,
                                  model_type=model_type)
#plt.show()
fout = open('./data/tmb2vec_data/metrics/'+'_dest_pre_score_series', 'wb')
pickle.dump(score_series1, fout)
fout = open('./data/tmb2vec_data/metrics/'  + '_dest_pre_results', 'wb')
pickle.dump(results1, fout)