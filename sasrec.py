import npu_support
import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from utility import pad_history,calculate_hit,extract_axis_1
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from SASRecModules_ori import *
import random
import json
import copy
import ast
import wandb

logging.getLogger().setLevel(logging.INFO)

        

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='Goodreads_5',
                        help='Toys_and_Games, Goodreads, Industrial_and_Scientific, CDs_and_Vinyl')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=32,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=int, default=1,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-5,
                        help='l2 loss reg coef.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='dro alpha.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='for robust radius')
    parser.add_argument("--model", type=str, default="SASRec",
                        help='the model name, GRU, Caser, SASRec')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='dropout ')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument("--early_stop", type=int, default=20,
                        help='the epoch for early stop')
    parser.add_argument("--eval_num", type=int, default=1,
                        help='evaluate every eval_num epoch' )
    parser.add_argument("--seed", type=int, default=1,
                        help="the random seed")
    parser.add_argument("--result_json_path", type=str, default="./result_temp/temp.json")
    parser.add_argument("--sample_num", type=int, default = 65536)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="bce")
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GRU(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, gru_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output

    def forward_eval(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, len_states.cpu(), batch_first=True, enforce_sorted=False
        )
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)

        return supervised_output


class Caser(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
                 dropout_rate):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

    def forward_eval(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)
        
        return supervised_output

class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

    def forward(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        # state_hidden = extract_axis_1(ff_out, len_states - 1)
        indices = (len_states -1 ).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        state_hidden = torch.gather(ff_out, 1, indices)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        # state_hidden = extract_axis_1(ff_out, len_states - 1)
        indices = (len_states -1 ).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        state_hidden = torch.gather(ff_out, 1, indices)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output


def evaluate_games(model, test_data, device, topk, save_logits=False, eval_type="test"):

    def calculate_hit_games_cuda(prediction, topk_list, target, hit_all, ndcg_all):
        rank_list = (prediction.shape[1] - 1 - torch.argsort(torch.argsort(prediction)))
        target_rank = torch.gather(rank_list, 1, target.view(-1, 1)).view(-1).clone()
        ndcg_temp_full = 1 / torch.log2(target_rank + 2)
        for i, top_k in enumerate(topk_list):
            mask = (target_rank < top_k)
            mask = mask.float()
            recall_temp = mask.sum()
            ndcg_temp = (ndcg_temp_full * mask).sum()
            hit_all[i] += recall_temp.cpu().item()
            ndcg_all[i] += ndcg_temp.cpu().item()
        return hit_all, ndcg_all

    # def calculate_hit_games(sorted_list, topk, true_items, hit_list, ndcg_list):
    #     for i in range(len(topk)):
    #         rec_list = sorted_list[:, -topk[i]:]
    #         for j in range(len(true_items)):
    #             if true_items[j] in rec_list[j]:
    #                 rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
    #                 hit_list[i] += 1.0
    #                 ndcg_list[i] += 1.0 / np.log2(rank + 1)

    # eval_seqs=pd.read_pickle(os.path.join(data_directory, test_data))
    eval_seqs = pd.read_csv(os.path.join(data_directory_test, test_data))
    eval_seqs = eval_seqs[['history_item_id', 'item_id']]
    eval_seqs = eval_seqs.rename(columns={'history_item_id': 'seq', 'item_id': 'next'})
    # transform '[1,2,3]' to [1,2,3]
    eval_seqs['seq'] = eval_seqs['seq'].apply(ast.literal_eval)
    eval_seqs['len_seq'] = eval_seqs['seq'].apply(lambda x: len(x))

    # right padding
    eval_seqs['seq'] = eval_seqs['seq'].apply(lambda x: x + [item_num] * (seq_size - len(x)))

    batch_size=1024
    hit_all = []
    ndcg_all = []
    for i in topk:
        hit_all.append(0)
        ndcg_all.append(0)
    total_samples = len(eval_seqs)
    total_batch_num = int(total_samples/batch_size) + (total_samples > batch_size * int(total_samples/batch_size))
    sasrec_logits = []
    for i in range(total_batch_num):
        begin = i * batch_size
        end = (i + 1) * batch_size
        if end > total_samples:
            batch = eval_seqs[begin:]
        else:
            batch = eval_seqs[begin:end]

        seq = list(batch['seq'].tolist())
        len_seq = list(batch['len_seq'])
        target=list(batch['next'])

        seq = torch.LongTensor(seq)

        seq = seq.to(device)

        target = torch.LongTensor(target).to(device)

        _ = model.eval()
        with torch.no_grad():
            prediction = model.forward_eval(seq, torch.tensor(np.array(len_seq)).to(device))
            sasrec_logits.append(prediction)
            # # print(prediction)
            # prediction = prediction.cpu()
            # prediction = prediction.detach().numpy()
            # print(prediction)
            # prediction=sess.run(GRUnet.output, feed_dict={GRUnet.inputs: states,GRUnet.len_state:len_states,GRUnet.keep_prob:1.0})
            # sorted_list=np.argsort(prediction)
            hit_all, ndcg_all = calculate_hit_games_cuda(prediction,topk, target, hit_all, ndcg_all)
    print('#############################################################')
    # logging.info('#############################################################')
    # print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    # logging.info('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    sasrec_logits = torch.cat(sasrec_logits, dim=0)
    # save sasrec_logits as npy file
    if save_logits and args.model == "SASRec":
        # torch.save(sasrec_logits, f"./code/baselines/result_temp/{args.data}_{args.model}_emb{args.hidden_factor}_bs{args.batch_size}_lr{args.lr}_decay{args.l2_decay}_seed{args.seed}_logits.npy")
        np.save(f"./result_temp/{args.data}_{args.model}_emb{args.hidden_factor}_bs{args.batch_size}_lr{args.lr}_decay{args.l2_decay}_seed{args.seed}_loss_{args.loss_type}_dropout{args.dropout_rate}_logits.npy", sasrec_logits.detach().cpu().numpy())
    hr_list = []
    ndcg_list = []
    # logging.info('#############################################################')
    for i in range(len(topk)):
        hr_purchase=hit_all[i]/len(eval_seqs)
        ng_purchase=ndcg_all[i]/len(eval_seqs)

        hr_list.append(hr_purchase)
        try:
            ndcg_list.append(ng_purchase)
        except:
            if ng_purchase == 0:
                ndcg_list.append(0)
            else:
                return "error"

    ndcg_last = ndcg_list[-1]

    str1 = ''
    str2 = ''
    for i in range(len(topk)):
        if eval_type == "test":
            str1 += 'hr@{}\tndcg@{}\t'.format(topk[i], topk[i])
            str2 += '{:.6f}\t{:.6f}\t'.format(hr_list[i], ndcg_list[i])
            wandb.log({
            f'Recall@{topk[i]}': hr_list[i],
            f'NDCG@{topk[i]}': ndcg_list[i]
            })

    print(str1)
    print(str2)
    print('#############################################################')
    if eval_type == "test":
        metrics_dict = {f'HR@{topk[i]}': hr_list[i] for i in range(len(topk))}
        metrics_dict.update({f'NDCG@{topk[i]}': ndcg_list[i] for i in range(len(topk))})
        wandb.log(metrics_dict)


    return ndcg_last, hr_list, ndcg_list


def calcu_propensity_score(buffer):
    items = list(buffer['next'])
    freq = Counter(items)
    for i in range(item_num):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(item_num)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.05)
    return ps

class RecDataset(Dataset):
    def __init__(self, data_df):
        self.data = data_df

    def __getitem__(self, i):
        temp = self.data.iloc[i]
        seq = torch.tensor(temp['seq'])
        len_seq = torch.tensor(temp['len_seq'])
        next = torch.tensor(temp['next'])
        return seq, len_seq, next

    def __len__(self):
        return len(self.data)

def main(topk, data_file_train, data_file_test, data_file_valid):
    if not args.debug:
        run = wandb.init(
            project="Rec",
            name=(
                f"{args.data}_{args.model}_emb{args.hidden_factor}_bs{args.batch_size}_lr{args.lr}_decay{args.l2_decay}_seed{args.seed}_loss_{args.loss_type}_dropout{args.dropout_rate}"
            ),  # Set the run name directly in the `init` method
            config={  # You can add your configuration here if needed
                "data": args.data,
                "model": args.model,
                "hidden_factor": args.hidden_factor,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "loss_type": args.loss_type,
            },
        )
        wandb.run.name = run.name
    else:
        os.environ["WANDB_DISABLED"] = "true"



    if args.model=='SASRec':
        model = SASRec(args.hidden_factor, item_num, seq_size, args.dropout_rate, device)
    if args.model=="GRU":
        model = GRU(args.hidden_factor,item_num, seq_size)
    if args.model=="Caser":
        model = Caser(args.hidden_factor,item_num, seq_size, args.num_filters, args.filter_sizes, args.dropout_rate)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    if args.loss_type == "bce":
        model_loss = nn.BCEWithLogitsLoss()
    elif args.loss_type == "ce":
        model_loss = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss type: {args.loss_type}")

    model.to(device)

    train = pd.read_csv(os.path.join(data_directory_train, data_file_train))
    train = train[['history_item_id', 'item_id']]
    train = train.rename(columns={'history_item_id': 'seq', 'item_id': 'next'})
    # transform '[1,2,3]' to [1,2,3]
    train['seq'] = train['seq'].apply(ast.literal_eval)
    train['len_seq'] = train['seq'].apply(lambda x: len(x))

    # right padding
    train['seq'] = train['seq'].apply(lambda x: x + [item_num] * (seq_size - len(x)))

    # train_data_org = train
    # train_data = train_data_org.sample(n=args.sample_num ,random_state=args.seed)
    train_data = train
    

    train_dataset = RecDataset(train_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    ps = calcu_propensity_score(train_data)
    ps = torch.tensor(ps)
    ps = ps.to(device)

    total_step=0
    ndcg_max = 0
    best_epoch = 0

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size) + (int(num_rows/args.batch_size) * args.batch_size != num_rows)
    for i in range(args.epoch):
        # for j in tqdm(range(num_batches)):
        for j, (seq, len_seq, target) in tqdm(enumerate(train_loader)):
            target_neg = []
            for index in range(len(len_seq)):
                neg=np.random.randint(item_num)
                while neg==target[index]:
                    neg = np.random.randint(item_num)
                target_neg.append(neg)
            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)
            target_neg = torch.LongTensor(target_neg)
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            target_neg = target_neg.to(device)

            if args.model=="GRU":
                len_seq = len_seq.cpu()

            model_output = model.forward(seq, len_seq)


            target = target.view((-1, 1))
            target_neg = target_neg.view((-1, 1))

            pos_scores = torch.gather(model_output, 1, target)
            neg_scores = torch.gather(model_output, 1, target_neg)

            pos_labels = torch.ones((len(len_seq), 1))
            neg_labels = torch.zeros((len(len_seq), 1))

            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            labels = labels.to(device)

            if args.loss_type == "bce":
                loss = model_loss(scores, labels)
            elif args.loss_type == "ce":
                loss = model_loss(model_output, target.squeeze(-1).long())
            else:
                raise ValueError(f"Invalid loss type: {args.loss_type}")


            pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
            pos_scores_dro = torch.squeeze(pos_scores_dro)
            pos_loss_dro = torch.gather(torch.mul((model_output - 1) * (model_output - 1), ps), 1, target)
            pos_loss_dro = torch.squeeze(pos_loss_dro)

            inner_dro = torch.sum(torch.exp((torch.mul(model_output * model_output, ps) / args.beta)), 1) - torch.exp((pos_scores_dro / args.beta)) + torch.exp((pos_loss_dro / args.beta)) 


            loss_dro = torch.log(inner_dro + 1e-24)
            if args.alpha == 0.0:
                loss_all = loss
            else:
                loss_all = loss + args.alpha * torch.mean(loss_dro)
            loss_all.backward()
            optimizer.step()

            if True:

                total_step+=1


                if total_step % (num_batches * args.eval_num) == 0:
                        print('VAL PHRASE:')
                        ndcg_last, val_hr, val_ndcg = evaluate_games(model, data_file_valid, device, topk, eval_type="val")
                        # ndcg_last, val_hr, val_ndcg = evaluate_games_old(model, 'val_sessions.df', device, topk)
                        print('TEST PHRASE:')
                        _, test_hr, test_ndcg = evaluate_games(model, data_file_test, device, topk, eval_type="test")

                        model = model.train()

                        if ndcg_last > ndcg_max:

                            ndcg_max = ndcg_last
                            best_epoch = i
                            early_stop = 0
                            best_hr = val_hr
                            best_ndcg = val_ndcg
                            best_model = copy.deepcopy(model)
                        
                        else:
                            early_stop += 1
                            if early_stop > args.early_stop:
                                return best_model, best_ndcg, best_hr
                        
                        print('BEST EPOCH:{}'.format(best_epoch))
                        print('EARLY STOP:{}'.format(early_stop))
                        print("best hr:")
                        print(best_hr)
                        print("best ndcg")
                        print(best_ndcg)
    return best_model, best_ndcg, best_hr
    


if __name__ == '__main__':
    topk=[1,3,5,10,20]
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    setup_seed(args.seed)

    data_directory_train = './data/Amazon/train/'
    data_directory_test = './data/Amazon/test/'
    data_directory_valid = './data/Amazon/valid/' 
    data_directory_info = './data/Amazon/info/'
    # data = pd.read_csv(data_directory + '/train/train.csv')
    # find the one csv file with arg.data in the name
    data_file_train = [f for f in os.listdir(data_directory_train) if args.data in f and f.endswith('.csv')]
    data_file_test = [f for f in os.listdir(data_directory_test) if args.data in f and f.endswith('.csv')]
    data_file_valid = [f for f in os.listdir(data_directory_valid) if args.data in f and f.endswith('.csv')]
    data_file_info = [f for f in os.listdir(data_directory_info) if args.data in f and f.endswith('.txt')]

    print(data_file_train)

    assert len(data_file_train) == 1, "There should be only one csv file with the name containing " + args.data
    assert len(data_file_test) == 1, "There should be only one csv file with the name containing " + args.data
    assert len(data_file_valid) == 1, "There should be only one csv file with the name containing " + args.data
    assert len(data_file_info) == 1, "There should be only one txt file with the name containing " + args.data
    data_file_train = data_file_train[0]
    data_file_test = data_file_test[0]
    data_file_valid = data_file_valid[0]
    data_file_info = data_file_info[0]

    with open(os.path.join(data_directory_info, data_file_info), 'r') as f:
        info = f.readlines()
        info = ["\"" + _.split('\t')[0].strip(' ') + "\"\n" for _ in info]
        data_info = info

    # data_info = pd.read_csv(os.path.join(data_directory_info, data_file_info))
    seq_size = 10  # the length of history to define the seq
    item_num = len(data_info)  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model, test_ndcg, test_hr = main(topk, data_file_train, data_file_test, data_file_valid)
    # temp = main(topk)
    result_dict = {}
    result_dict["NDCG"] = {}
    result_dict["HR"] = {}
    for i,k in enumerate(topk):
        result_dict["NDCG"][k] = test_ndcg[i]
        result_dict["HR"][k] = test_hr[i]

    result_folder = ""
    for path_name in args.result_json_path.split("/")[:-1]:
        result_folder += path_name + "/"

    os.makedirs(result_folder, exist_ok=True)
    with open(args.result_json_path ,'w',encoding='utf-8') as f:
        json.dump(result_dict, f,ensure_ascii=False, indent=1)
    # torch.save(best_model, result_folder + f"/best_{args.data}_{args.model}_emb{args.hidden_factor}_bs{args.batch_size}_lr{args.lr}_decay{args.l2_decay}_seed{args.seed}_loss_{args.loss_type}")
    torch.save(best_model.state_dict(), result_folder + f"/best_{args.data}_{args.model}_emb{args.hidden_factor}_bs{args.batch_size}_lr{args.lr}_decay{args.l2_decay}_seed{args.seed}_loss_{args.loss_type}_dropout{args.dropout_rate}_state.pth")

    evaluate_games(best_model, data_file_test, device, topk, save_logits=True, eval_type="test")

