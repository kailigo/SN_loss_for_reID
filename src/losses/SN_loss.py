# coding=utf-8
from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class SN_LOSS(nn.Module):
    def __init__(self, alpha=30, margin=1, k=16, weight=1.0):
        super(SN_LOSS, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.K = k
        # self.r = r
        self.weight = weight

        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, inputs, targets):

        n = inputs.size(0)
        # Compute pairwise distance
        # dist_mat = re_ranking_dist(inputs)
        dist_mat = euclidean_dist(inputs)

        # dist_mat = re_ranking_retrieval(dist_mat, k1=20, k2=6, lambda_value=0.3)

        targets = targets.cuda()
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).cuda()
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        #

        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_dist = torch.masked_select(dist_mat, pos_mask)
        neg_dist = torch.masked_select(dist_mat, neg_mask)

        num_instances = len(pos_dist)//n + 1
        num_neg_instances = n - num_instances

        pos_dist = pos_dist.resize(len(pos_dist)//(num_instances-1), num_instances-1)
        neg_dist = neg_dist.resize(
            len(neg_dist) // num_neg_instances, num_neg_instances)

        loss = list()
        acc_num = 0


        # 遍历Anchor, 每个样本都作为Anchor,来计算损失
        for i, pos_pair in enumerate(pos_dist):
            # pos_pair是以第i个样本为Anchor的所有正样本的距离
            pos_pair = torch.sort(pos_pair)[0]
            # neg_pair是以第i个样本为Anchor的所有负样本的距离
            # neg_pair = neg_dist[i] * neg_dist[i]

            neg_pair = neg_dist[i]
            pair = torch.cat([pos_pair, neg_pair])
            threshold = torch.sort(pair)[0][self.K]

            pos_neig = torch.masked_select(pos_pair, pos_pair < threshold)
            neg_neig = torch.masked_select(neg_pair, neg_pair < threshold)

            # neg_pair = neg_dist[i]
            # pos_neig = torch.sort(pos_pair)[0]
            # neg_neig = torch.sort(neg_pair)[0][:len(pos_neig)]

            # import pdb
            # pdb.set_trace()

            # neg_neig = neg_pair[:self.K]

            # 第K+1个近邻点到Anchor的距离值
            # pair = torch.cat([pos_pair, neg_pair])
            # threshold = torch.sort(pair)[0][self.K]

            # 取出K近邻中的正样本对和负样本对
            # pos_neig = torch.masked_select(pos_pair, pos_pair < threshold)
            # neg_neig = torch.masked_select(neg_pair, neg_pair < threshold)

            # 若前K个近邻中没有正样本，则仅取最近正样本
            if len(pos_neig) == 0:
                pos_neig = pos_pair[0]

            # if ~math.isfinite(pos_neig):
            #    import pdb
            #    pdb.set_trace()

            # if i == 1 and np.random.randint(64) == 1:
            #     print('pos_pair is ---------', pos_neig)
            #     print('neg_pair is ---------', neg_neig)

            # 计算logit, 1 的作用是防止超过计算机浮点数
            pos_logit = torch.sum(torch.exp(self.alpha * (1 - pos_neig)))
            neg_logit = torch.sum(torch.exp(self.alpha * (1 - neg_neig)))

            knn_loss = -torch.log(pos_logit / (pos_logit + neg_logit))

            # A = 0.85210
            # A = numpy.array([A])

            if len(pos_neig) == 1:
                # cont_loss = Variable(0.0)
                cont_loss = pos_neig[0]
                # cont_loss = Variable(torch.from_numpy(np.array([0.0])))
            else:
                cont_loss = pos_neig[-1] - pos_neig[0]

            # knn_loss = 0
            cont_loss = pos_neig[0]

            # cont_loss = pos_neig[-1]
            # torch.clamp(self.r - pos_neig[-1], min=0)
            loss_ = knn_loss + self.weight*cont_loss

            # print(pos_logit)
            # print(neg_logit)
            # print(loss_)
            # import pdb
            # pdb.set_trace()

            if loss_.data[0] < 0.6:
                acc_num += 1
            loss.append(loss_)

            # print('[Epoch %05d: step %05d]\t Loss: %.6f \t Accuracy: %.3f \t Pos-Dist: %.3f \t Neg-Dist: %.3f'
            #       % (ep + 1, step + 1, loss.data[0], inter_, dist_ap, dist_an))

            # print('knn_loss: %.3f \t cont_loss: %.3f' % (knn_loss.data[0], cont_loss.data[0]))
            # print('cont_loss is ---------', cont_loss)


        # 遍历所有样本为Anchor，对Loss取平均
        loss1 = torch.mean(torch.cat(loss))
        loss = loss1

        accuracy = float(acc_num)/n
        neg_d = torch.mean(neg_dist).data[0]
        pos_d = torch.mean(pos_dist).data[0]

        return loss, accuracy, pos_d, neg_d


def re_ranking_retrieval(orignal_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    # original_dist = np.concatenate(
    #   [np.concatenate([q_q_dist, q_g_dist], axis=1),
    #    np.concatenate([q_g_dist.T, g_g_dist], axis=1)], axis=0)

    original_dist = np.power(orignal_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    # query_num = q_g_dist.shape[0]
    gallery_num = original_dist.shape[0]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    # original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(all_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    # final_dist = final_dist[:query_num,query_num:]
    return final_dist




def euclidean_dist(inputs_):
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    # for numerical stability
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))
    # targets = targets

    inputs = inputs.cuda()
    print(myKNNSoftmax(alpha=30)(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
