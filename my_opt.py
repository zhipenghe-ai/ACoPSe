"""Train and evaluate the paired-feature ACoPSe hashing models."""

import os

import torch
import torch.nn.functional as F

from load_data import get_loader_flickr, get_loader_nus, get_loader_coco
from torch.autograd import Variable
from models import MyNet, Img_Net, Txt_Net
from utils import compress, calculate_top_map
import numpy as np
from torch_clustering import PyTorchKMeans
import lightly.loss as DLC_loss

RETRIEVAL_TOP_K = 5000


class PCIRH:
    def __init__(self, log, config):
        self.config = config
        self.log = log

        data_dir = getattr(config, 'DATA_DIR', './datasets')
        if config.DATASET == 'MIRFlickr':
            dataloader, data_train = get_loader_flickr(config.BATCH_SIZE, data_dir=data_dir)
        elif config.DATASET == 'NUSWIDE':
            dataloader, data_train = get_loader_nus(config.BATCH_SIZE, data_dir=data_dir)
        elif config.DATASET == 'COCO':
            dataloader, data_train = get_loader_coco(config.BATCH_SIZE, data_dir=data_dir)

        self.I_tr, self.T_tr, self.L_tr = data_train

        self.train_num = self.I_tr.shape[0]
        self.img_feat_len = self.I_tr.shape[1]
        self.txt_feat_len = self.T_tr.shape[1]

        self.train_loader = dataloader['train']
        self.test_loader = dataloader['query']
        self.database_loader = dataloader['database']

        img_norm = F.normalize(torch.Tensor(self.I_tr)).cuda()
        txt_norm = F.normalize(torch.Tensor(self.T_tr)).cuda()
        self.S_cmm = self.cal_similarity(img_norm, txt_norm)

        self.mynet = MyNet(code_len=self.config.HASH_BIT, ori_featI=self.img_feat_len, ori_featT=self.txt_feat_len).cuda()
        self.imgnet = Img_Net(code_len=self.config.HASH_BIT, img_feat_len=self.img_feat_len).cuda()
        self.txtnet = Txt_Net(code_len=self.config.HASH_BIT, txt_feat_len=self.txt_feat_len).cuda()

        self.opt_mynet = torch.optim.Adam(self.mynet.parameters(), lr=config.LR_MyNet)
        self.opt_imgnet = torch.optim.Adam(self.imgnet.parameters(), lr=config.LR_IMG)
        self.opt_txtnet = torch.optim.Adam(self.txtnet.parameters(), lr=config.LR_TXT)

        self.record_Lsmodel = []
        self.record_Lshfunc = []
        self.official_DLC = DLC_loss.DCLLoss(self.config.temperature)

    def train_method(self, epoch):
        cluster_num = self.config.cluster_num
        if not 2 <= cluster_num <= self.train_num:
            raise ValueError(f'cluster_num must be between 2 and {self.train_num}, got {cluster_num}')

        coll_B = list([])
        record_index = list([])
        Ls_method = 0

        self.mynet.train()

        psedo_labels = self.psedo_labeling()
        for No, (img, txt, _, index_) in enumerate(self.train_loader):

            img = Variable(img.cuda().to(torch.float))
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())

            S_batch = self.S_cmm[index_, :][:, index_].cuda()

            imgN = img
            txtN = txt

            HI, HT, H, B, DeI_feat, DeT_feat,VI,VT,VgcnI,VgcnT,fusion,HJ = self.mynet(imgN, txtN, S_batch)

            coll_B.extend(B.cpu().data.numpy())
            record_index.extend(index_)

            self.opt_mynet.zero_grad()

            loss_CCR = self.config.lambda1 * (F.mse_loss(DeI_feat, imgN) + F.mse_loss(DeT_feat, txtN))
            loss_2 = self.loss_method(HI, HT, H, B, S_batch)

            B_psedo_labels = psedo_labels[index_]
            img_centers = self.compute_centers(x=VI, psedo_labels=B_psedo_labels, num_cluster=cluster_num)
            txt_centers = self.compute_centers(x=VT, psedo_labels=B_psedo_labels, num_cluster=cluster_num)
            cluster_loss = self.config.lambda3 * self.compute_cluster_loss(
                img_centers, txt_centers, self.config.k_means, B_psedo_labels, num_cluster=cluster_num
            )

            dcl_loss = self.official_DLC(VI, fusion) + self.official_DLC(VT, fusion)
            loss_model = loss_CCR + loss_2 + dcl_loss + cluster_loss
            Ls_method = (Ls_method + loss_model).item()

            loss_model.backward()
            self.opt_mynet.step()

            if (No + 1) % (self.train_num // self.config.BATCH_SIZE / self.config.EPOCH_INTERVAL) == 0:
                self.log.info('Epoch [%d/%d], Iter [%d/%d] Ls1=%.4f, Ls2=%.4f, Ls_model: %.4f,dcl_loss: %.4f,'
                                 % (epoch + 1, self.config.NUM_EPOCH, No + 1, self.train_num // self.config.BATCH_SIZE,
                                     loss_CCR, loss_2, loss_model.item(),dcl_loss))

        coll_B = np.array(coll_B)

        self.record_Lsmodel.append(Ls_method)

        return coll_B, record_index

    def train_Hashfunc(self, coll_B, record_index, epoch):

        self.imgnet.train()
        self.txtnet.train()

        Ls_hfunc = 0

        B = torch.from_numpy(coll_B).cuda()

        img = torch.Tensor(self.I_tr[record_index,:]).cuda()
        txt = torch.Tensor(self.T_tr[record_index,:]).cuda()

        img_norm = F.normalize(img)
        txt_norm = F.normalize(txt)

        num_cyc = img_norm.shape[0] / self.config.BATCH_SIZE
        num_cyc = int(num_cyc+1) if num_cyc-int(num_cyc)>0 else int(num_cyc)

        for kk in range(num_cyc):
            if kk+1 < num_cyc:
                img_batch = img_norm[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
                txt_batch = txt_norm[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
                B_batch = B[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
                S_batch = self.S_cmm[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE,
                                   kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE]
            else:
                img_batch = img_norm[kk * self.config.BATCH_SIZE:, :]
                txt_batch = txt_norm[kk * self.config.BATCH_SIZE:, :]
                B_batch = B[kk * self.config.BATCH_SIZE:, :]
                S_batch = self.S_cmm[kk * self.config.BATCH_SIZE:, kk * self.config.BATCH_SIZE:]

            hfunc_BI = self.imgnet(img_batch)
            hfunc_BT = self.txtnet(txt_batch)

            self.opt_imgnet.zero_grad()
            self.opt_txtnet.zero_grad()

            loss_f1 = F.mse_loss(hfunc_BI, B_batch) + F.mse_loss(hfunc_BT, B_batch) + F.mse_loss(hfunc_BI, hfunc_BT)

            S_BI_BT = F.normalize(hfunc_BI).mm(F.normalize(hfunc_BT).t())
            S_BI_BI = F.normalize(hfunc_BI).mm(F.normalize(hfunc_BI).t())
            S_BT_BT = F.normalize(hfunc_BT).mm(F.normalize(hfunc_BT).t())
            loss_f2 = F.mse_loss(S_BI_BT, S_batch) + F.mse_loss(S_BI_BI, S_batch) + F.mse_loss(S_BT_BT, S_batch)

            f3 = self.cross_modal_similarity_loss(hfunc_BI, hfunc_BT)
            loss_hfunc = loss_f1 + self.config.beta * f3
            Ls_hfunc = (Ls_hfunc + loss_hfunc).item()

            loss_hfunc.backward()

            self.opt_imgnet.step()
            self.opt_txtnet.step()

        self.log.info('Epoch [%d/%d], Ls_hfunc: %.4f' % (epoch + 1, self.config.NUM_EPOCH, loss_hfunc.item()))

        self.record_Lshfunc.append(Ls_hfunc)

        return self.imgnet, self.txtnet

    def performance_eval(self):

        self.log.info('Evaluation: MAP@%d', RETRIEVAL_TOP_K)
        self.imgnet.eval().cuda()
        self.txtnet.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.imgnet, self.txtnet)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=RETRIEVAL_TOP_K)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=RETRIEVAL_TOP_K)
        self.log.info('MAP@%d I->T: %.3f, MAP@%d T->I: %.3f',
                      RETRIEVAL_TOP_K, MAP_I2T, RETRIEVAL_TOP_K, MAP_T2I)

        return MAP_I2T, MAP_T2I

    def eval2(self):

        self.log.info('Evaluation: MAP@50')
        self.imgnet.eval().cuda()
        self.txtnet.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.imgnet, self.txtnet)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        return MAP_I2T, MAP_T2I

    def NN_select(self, S):

        m, n1 = S.sort()
        ind1 = n1[:, -4:-1][:,0]
        ind2 = n1[:, -4:-1][:,1]

        return ind1, ind2

    def cal_similarity(self, F_I, F_T):
        batch_size = F_I.size(0)
        size = batch_size
        top_size = self.config.K

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())

        S1 = self.config.a1 * S_I + (1 - self.config.a1) * S_T

        m, n1 = S1.sort()
        S1[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(-1)] = 0.

        S2 = 2.0 / (1 + torch.exp(-S1)) - 1 + torch.eye(S1.size(0)).cuda()
        S2 = (S2 + S2.t())/2
        S = self.config.a2 * S1 + (1 - self.config.a2) * S2

        return S

    def loss_method(self, HI, HT, H, B, S_batch):

        HI_norm = F.normalize(HI)
        HT_norm = F.normalize(HT)
        H_norm = F.normalize(H)

        HI_HI = HI_norm.mm(HI_norm.t())
        HT_HT = HT_norm.mm(HT_norm.t())
        H_H = H_norm.mm(H_norm.t())

        loss_ACR = self.config.lambda2 * (
                F.mse_loss(S_batch, HI_HI) + F.mse_loss(S_batch, HT_HT) + F.mse_loss(S_batch, H_H))
        loss_DIS = F.mse_loss(H, B)

        return loss_ACR + loss_DIS


    def _checkpoint_path(self, file_name=None):
        if file_name is None:
            file_name = f'{self.config.DATASET}_{self.config.HASH_BIT}bits.pth'
        return os.path.join(self.config.MODEL_DIR, file_name)

    def save_checkpoints(self):
        ckp_path = self._checkpoint_path()
        os.makedirs(self.config.MODEL_DIR or '.', exist_ok=True)
        obj = {
            'ImgNet': self.imgnet.state_dict(),
            'TxtNet': self.txtnet.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.log.info('**********Save the trained model successfully.**********')


    def load_checkpoints(self, file_name=None):
        ckp_path = self._checkpoint_path(file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.log.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.log.error('********** Fail to load checkpoint %s!*********' % ckp_path)
            raise

        self.imgnet.load_state_dict(obj['ImgNet'])
        self.txtnet.load_state_dict(obj['TxtNet'])

    def psedo_labeling(self):
        loader = self.train_loader
        img_list = []
        txt_list = []
        sample_indices = []
        for No, (images, texts, _, index_) in enumerate(loader):
            images = Variable(images.cuda().to(torch.float))
            texts = Variable(torch.FloatTensor(texts.numpy()).cuda())
            S_batch = self.S_cmm[index_, :][:, index_]
            with torch.no_grad():
                HI, HT, H, B, DeI_feat, DeT_feat, VI, VT,VgcnI,VgcnT,fusion,_=self.mynet(images,texts,S_batch)
                img_list.append(VI)
                txt_list.append(VT)
                sample_indices.append(index_)
        all_img = torch.cat(img_list, dim=0)
        all_txt = torch.cat(txt_list, dim=0)
        all = torch.cat((all_img, all_txt), dim=1)
        assignments = self.clustering(features=all)
        # K-means returns assignments in loader order. Training looks them up by dataset index.
        indices = torch.cat(sample_indices).to(device=assignments.device)
        labels_by_sample = torch.empty_like(assignments)
        labels_by_sample.index_copy_(0, indices, assignments)
        return labels_by_sample

    def compute_centers(self, x, psedo_labels, num_cluster=24):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(num_cluster, n_samples).to(x)  # L, N
            weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)
        return centers

    def compute_cluster_loss(self,
                             q_centers,
                             k_centers,
                             temperature,
                             psedo_labels,
                             num_cluster=24):
        d_q = q_centers.mm(q_centers.T) / temperature
        d_k = (q_centers * k_centers).sum(dim=1) / temperature
        d_q = d_q.float()
        d_q[torch.arange(num_cluster), torch.arange(num_cluster)] = d_k

        zero_classes = torch.arange(num_cluster).cuda()[torch.sum(F.one_hot(torch.unique(psedo_labels),
                                                                            num_cluster), dim=0) == 0]
        mask = torch.zeros((num_cluster, num_cluster), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((num_cluster, num_cluster))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, num_cluster - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(num_cluster, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (num_cluster - len(zero_classes))
        return loss

    def clustering(self, features):
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': self.config.cluster_num,
            'verbose': False
        }
        clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features)
        return psedo_labels


    def cross_modal_similarity_loss(sefl,HI, HT):
        HI_norm = torch.norm(HI, dim=1, keepdim=True)
        HT_norm = torch.norm(HT, dim=1, keepdim=True)

        dot_product = torch.sum(HI * HT, dim=1, keepdim=True)

        cosine_similarity = dot_product / (HI_norm * HT_norm)

        loss = 1 - torch.mean(cosine_similarity)

        return loss

    def warmup(self,epoch):
        for i in range (epoch):
            for No, (img, txt, _, index_) in enumerate(self.train_loader):

                img = Variable(img.cuda().to(torch.float))
                txt = Variable(torch.FloatTensor(txt.numpy()).cuda())

                S_batch = self.S_cmm[index_, :][:, index_].cuda()

                Iind1, Iind2 = self.NN_select(S_batch)
                Tind1, Tind2 = self.NN_select(S_batch)
                imgNN1, imgNN2 = img[Iind1, :], img[Iind2, :]
                txtNN1, txtNN2 = txt[Tind1, :], txt[Tind2, :]

                imgN = 0.7 * img + 0.2 * imgNN2 + 0.1 * imgNN1
                txtN = 0.7 * txt + 0.2 * txtNN2 + 0.1 * txtNN1

                HI, HT, H, B, DeI_feat, DeT_feat,VI,VT,VgcnI,VgcnT,_,_ = self.mynet(imgN, txtN, S_batch)

                self.opt_mynet.zero_grad()

                loss_CCR = self.config.lambda1 * (F.mse_loss(DeI_feat, imgN) + F.mse_loss(DeT_feat, txtN))

                loss_2 = self.loss_method(HI, HT, H, B, S_batch)


                loss_model =loss_CCR

                loss_model.backward()
                self.opt_mynet.step()

                if (No + 1) % (self.train_num // self.config.BATCH_SIZE / self.config.EPOCH_INTERVAL) == 0:
                    self.log.info('Epoch [%d/%d], Iter [%d/%d] Ls1=%.4f, Ls2=%.4f, Ls_model: %.4f'
                                     % (i + 1, epoch, No + 1, self.train_num // self.config.BATCH_SIZE,
                                         loss_CCR, loss_2, loss_model.item()))
