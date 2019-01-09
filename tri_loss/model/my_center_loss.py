import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # self.loss_weight = loss_weight

        self.centers = torch.zeros(num_classes, feat_dim)
        # self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        # self.register_parameter('centers', self.centers) # no need to register manually. See nn.Module.__setattr__(...)
        # self.use_cuda = False

    # def forward(self, y, feat):
    #     # torch.histc can only be implemented on CPU
    #     # To calculate the total number of every class in one mini-batch. See Equation 4 in the paper
    #     if self.use_cuda:
    #         hist = Variable(torch.histc(y.cpu().data.float(),bins=self.num_classes,min=0,max=self.num_classes) + 1).cuda()
    #     else:
    #         hist = Variable(torch.histc(y.data.float(),bins=self.num_classes,min=0,max=self.num_classes) + 1)
    #
    #     centers_count = hist.index_select(0,y.long())
    #
    #
    #     # To squeeze the Tenosr
    #     batch_size = feat.size()[0]
    #     feat = feat.view(batch_size, 1, 1, -1).squeeze()
    #     # To check the dim of centers and features
    #     if feat.size()[1] != self.feat_dim:
    #         raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size()[1]))
    #
    #     centers_pred = self.centers.index_select(0, y.long())
    #     diff = feat - centers_pred
    #     loss = self.loss_weight * 1 / 2.0 * (diff.pow(2).sum(1) / centers_count).sum()
    #     return loss
    #
    # def cuda(self, device_id=None):
    #     """Moves all model parameters and buffers to the GPU.
    #     Arguments:
    #         device_id (int, optional): if specified, all parameters will be
    #             copied to that device
    #     """
    #     self.use_cuda = False
    #     return self._apply(lambda t: t.cuda(device_id))

    def forward(self, features, target, alpha=1):
        batch_size = target.size(0)
        features_dim = features.size(1)
        centers = self.centers

        target_expand = target.view(batch_size,1).expand(batch_size,features_dim)
        centers_var = Variable(centers)
        centers_batch = centers_var.gather(0,target_expand.cpu())
        centers_batch = centers_batch.cuda()
        criterion = nn.MSELoss()
        center_loss = criterion(features,  centers_batch)

        diff = centers_batch - features
        unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
        appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
        appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)
        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
        diff_cpu = alpha * diff_cpu
        for i in range(batch_size):
            centers[target.data[i]] -= diff_cpu[i].type(centers.type())

        self.centers = centers

        # print(self.centers)

        return center_loss


# class CenterLoss(nn.Module):
#     def __init__(self, num_classes, feat_dim, lambda_c=1.0, use_cuda=True):
#         super(CenterLoss, self).__init__()
#         self.lambda_c = lambda_c
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim        
#         self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
#         self.use_cuda = use_cuda

#     def forward(self, y, feat):        
#         # batch_size = hidden.shape[0]
#         # expanded_centers = self.centers.index_select(dim=0, index=y)
#         # intra_distances = hidden.dist(expanded_centers)
#         # loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
#         # return loss

#         if self.use_cuda:
#             hist = Variable(torch.histc(y.cpu().data.float(),bins=self.num_classes,min=0,max=self.num_classes) + 1).cuda()
#         else:
#             hist = Variable(torch.histc(y.data.float(),bins=self.num_classes,min=0,max=self.num_classes) + 1)

#         centers_count = hist.index_select(0,y.long())
#         # To squeeze the Tenosr
#         batch_size = feat.size()[0]
#         feat = feat.view(batch_size, 1, 1, -1).squeeze()
#         # To check the dim of centers and features
#         if feat.size()[1] != self.feat_dim:
#             raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size()[1]))

#         centers_pred = self.centers.index_select(0, y.long())
#         diff = feat - centers_pred
#         loss = self.loss_weight * 1 / 2.0 * (diff.pow(2).sum(1) / centers_count).sum()
#         return loss


#     def cuda(self, device_id=None):
#         self.use_cuda = True
#         return self._apply(lambda t: t.cuda(device_id))



# class CenterLoss(nn.Module):
#     def __init__(self, dim_hidden, num_classes, lambda_c=1.0, use_cuda=True):
#         super(CenterLoss, self).__init__()
#         self.dim_hidden = dim_hidden
#         self.num_classes = num_classes
#         self.lambda_c = lambda_c
#         self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))
#         self.use_cuda = use_cuda

#     def forward(self, y, hidden):
#         batch_size = hidden.size()[0]
#         expanded_centers = self.centers.index_select(dim=0, index=y)
#         intra_distances = hidden.dist(expanded_centers)
#         loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
#         return loss

#     def cuda(self, device_id=None):
#         """Moves all model parameters and buffers to the GPU.

#         Arguments:
#             device_id (int, optional): if specified, all parameters will be
#                 copied to that device
#         """
#         self.use_cuda = True
#         return self._apply(lambda t: t.cuda(device_id))


# def test():
#     ct = CenterLoss(2, 2, use_cuda=False)
#     y = Variable(torch.LongTensor([0, 0, 0, 1]))
#     feat = Variable(torch.zeros(4, 2), requires_grad=True)

#     out = ct(y, feat)
#     out.backward()


# if __name__ == '__main__':
#     test()