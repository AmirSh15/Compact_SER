import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

import networkx as nx

from tqdm import tqdm

from utils.util import load_data, separate_data
from utils.radam import RAdam, AdamW
from models.graphcnn import Graph_CNN_ortega
from utils.pytorchtools import EarlyStopping


criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([1,1,1,1]))


def train(args, model, device, train_graphs, optimizer, epoch, A):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    start = time.time()
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss for Adj & Pool
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))
    end = time.time()
    print('epoch time: %f' %(end-start))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    ind = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, num_class):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred_ = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred_.eq(labels.view_as(pred_)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))


    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test, output, labels

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='Implementation of COMPACT GRAPH ARCHITECTURE FOR SPEECH EMOTION RECOGNITION paper')
    parser.add_argument('--dataset', type=str, default="IEMOCAP",
                        help='name of dataset (default: IEMOCAP)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 90)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=5,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling over nodes in a graph to get graph embeddig: sum or average')
    parser.add_argument('--graph_type', type=str, default="line", choices=["line", "cycle"],
                        help='Graph construction options')
    parser.add_argument('--Normalize', type=bool, default=True, choices=[True, False],
                        help='Normalizing data')
    parser.add_argument('--patience', type=int, default=10,
                        help='Normalizing data')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='beta2 for adam')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    ##load data
    graphs, num_classes = load_data(args.dataset, args.Normalize)

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    A = nx.to_numpy_matrix(train_graphs[0][0].g)
    if(args.graph_type == 'cycle'):
        A[0, -1] = 1
        A[-1, 0] = 1
    A = torch.Tensor(A).to(device)

    model = Graph_CNN_ortega(args.num_layers, train_graphs[0][0].node_features.shape[1],
                            args.hidden_dim, num_classes, args.final_dropout, args.graph_pooling_type,
                            device, A).to(device)

    Num_Param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    b = 0
    for p in model.parameters():
        if p.requires_grad:
            a = p.numel()
            b += a
    print("Number of Trainable Parameters= %d" % (Num_Param))

    acc_train_sum = 0
    acc_test_sum = 0

    for i in range(args.fold_idx):
        train_data =  train_graphs[i]
        test_data = test_graphs[i]

        # optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        #                   weight_decay=args.weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        #                   weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        for epoch in range(1, args.epochs + 1):
            scheduler.step()

            avg_loss = train(args, model, device, train_data, optimizer, epoch, A)

            if(epoch>1):
                #### Validation check
                with torch.no_grad():
                    val_out = pass_data_iteratively(model, test_data)
                    val_labels = torch.LongTensor([graph.label for graph in test_data]).to(device)
                    val_loss = criterion(val_out, val_labels)
                    val_loss = np.average(val_loss.detach().cpu().numpy())

                #### Check early stopping
                early_stopping(val_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break


            if((epoch>300)and(epoch % 20 ==0)) or (epoch % 10 ==0):
                acc_train, acc_test, _, _ = test(args, model, device, train_data, test_data, num_classes)


        model.load_state_dict(torch.load('checkpoint.pt'))

        acc_train, acc_test, output, label = test(args, model, device, train_data, test_data, num_classes)
        acc_train_sum += acc_train
        acc_test_sum += acc_test

        model = Graph_CNN_ortega(args.num_layers, train_graphs[0][0].node_features.shape[1],
                                 args.hidden_dim, num_classes, args.final_dropout, args.graph_pooling_type,
                                 device, A).to(device)

    print(
        'Average train acc: %f,  Average test acc: %f' % (acc_train_sum / args.fold_idx, acc_test_sum / args.fold_idx))


if __name__ == '__main__':
    main()
