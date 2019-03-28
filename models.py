import numpy as np
import torch
from torch import optim
from sklearn.neighbors import NearestNeighbors


class LocalColorTransfer:
    def __init__(self, s, g, featS_norm, featG_norm, kmeans_labels, device, kmeans_ratio=1, patch_size=3):
        self.source = torch.from_numpy(s).float().to(device)
        self.guide = torch.from_numpy(g).float().to(device)
        self.featS_norm = torch.from_numpy(featS_norm).float().to(device)
        self.featG_norm = torch.from_numpy(featG_norm).float().to(device)
        self.height = s.shape[0]
        self.width = s.shape[1]
        self.channel = s.shape[2]
        self.patch_size = patch_size

        self.paramA = torch.zeros(s.shape).to(device)
        self.paramB = torch.zeros(s.shape).to(device)
        self.sub = torch.ones(*s.shape[:2], 1).to(device)

        self.kmeans_labels = np.zeros(s.shape[:2]).astype(np.int32)
        self.kmeans_ratio = kmeans_ratio

        self.init_params(kmeans_labels)

    def init_params(self, kmeans_labels):
        """
            Initialize a and b from source and guidance using mean and std
        """
        eps = 0.002
        for y in range(self.height):
            for x in range(self.width):
                dy0 = dx0 = self.patch_size // 2
                dy1 = dx1 = self.patch_size // 2 + 1
                dy0 = min(y, dy0)
                dy1 = min(self.height - y, dy1)
                dx0 = min(x, dx0)
                dx1 = min(self.width - x, dx1)

                patchS = self.source[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)
                patchG = self.guide[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)
                self.paramA[y, x] = patchG.std(0) / (patchS.std(0) + eps)
                self.paramB[y, x] = patchG.mean(0) - self.paramA[y, x] * patchS.mean(0)
                self.sub[y, x, 0] += self.patch_size ** 2 - (dy0 + dy1) * (dx0 + dx1)

                y_adj = min(y // self.kmeans_ratio, kmeans_labels.shape[0] - 1)
                x_adj = min(x // self.kmeans_ratio, kmeans_labels.shape[1] - 1)
                self.kmeans_labels[y, x] = kmeans_labels[y_adj, x_adj]
        self.paramA.requires_grad_()
        self.paramB.requires_grad_()

    def visualize(self):
        transfered = self.paramA * self.source + self.paramB
        # imshow(transfered.data.cpu().numpy().astype(np.float64))
        # imshow(color.lab2rgb(transfered.data.cpu().numpy().astype(np.float64)))

    def loss_d(self):
        error = torch.pow(self.featS_norm - self.featG_norm, 2).sum(2)
        transfered = self.paramA * self.source + self.paramB
        term1 = 1 - error / 4
        term2 = torch.pow(transfered - self.guide, 2).sum(2)
        loss_d = torch.mean(term1 * term2)

        return loss_d

    def loss_l(self):
        patchS_stack = self.source.unsqueeze(2).repeat(1, 1, self.patch_size ** 2, 1)  # (self.height, self.width, 9, self.channel)
        patchA_stack = self.paramA.unsqueeze(2).repeat(1, 1, self.patch_size ** 2, 1)
        patchB_stack = self.paramB.unsqueeze(2).repeat(1, 1, self.patch_size ** 2, 1)
        for y in range(self.height):
            for x in range(self.width):
                dy0 = dx0 = self.patch_size // 2
                dy1 = dx1 = self.patch_size // 2 + 1
                dy0 = min(y, dy0)
                dy1 = min(self.height - y, dy1)
                dx0 = min(x, dx0)
                dx1 = min(self.width - x, dx1)

                patchS_stack[y, x, :((dy0 + dy1) * (dx0 + dx1))] = self.source[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)
                patchA_stack[y, x, :((dy0 + dy1) * (dx0 + dx1))] = self.paramA[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)
                patchB_stack[y, x, :((dy0 + dy1) * (dx0 + dx1))] = self.paramB[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)

        patchSD = torch.norm(self.source.unsqueeze(2) - patchS_stack, 2, 3).exp()
        wgt = patchSD / (patchSD.sum(2, keepdim=True) - self.sub)
        # Getting norm term
        term1 = torch.pow(self.paramA.unsqueeze(2) - patchA_stack, 2).sum(3)
        term2 = torch.pow(self.paramB.unsqueeze(2) - patchB_stack, 2).sum(3)
        term3 = term1 + term2
        loss_l = torch.sum(wgt * term3, 2).mean()

        return loss_l

    def loss_nl(self):
        patchS_stack = list()
        patchA_stack = list()
        patchB_stack = list()
        mixedS = list()
        mixedA = list()
        mixedB = list()

        index_map = np.zeros((2, self.height, self.width)).astype(np.int32)
        index_map[0] = np.arange(self.height)[:, np.newaxis] + np.zeros(self.width).astype(np.int32)
        index_map[1] = np.zeros(self.height).astype(np.int32)[:, np.newaxis] + np.arange(self.width)

        for i in range(5):
            index_map_cluster = index_map[:, self.kmeans_labels == i]
            source_cluster = self.source[index_map_cluster[0], index_map_cluster[1]]
            paramA_cluster = self.paramA[index_map_cluster[0], index_map_cluster[1]]
            paramB_cluster = self.paramB[index_map_cluster[0], index_map_cluster[1]]

            nbrs = NearestNeighbors(n_neighbors=9, n_jobs=1).fit(source_cluster)
            indices = nbrs.kneighbors(source_cluster, return_distance=False)

            patchS_stack.append(source_cluster[indices[:, 1:].reshape(-1)].reshape(-1, 8, self.channel))
            patchA_stack.append(paramA_cluster[indices[:, 1:].reshape(-1)].reshape(-1, 8, self.channel))
            patchB_stack.append(paramB_cluster[indices[:, 1:].reshape(-1)].reshape(-1, 8, self.channel))
            mixedS.append(source_cluster.unsqueeze(1))
            mixedA.append(paramA_cluster.unsqueeze(1))
            mixedB.append(paramB_cluster.unsqueeze(1))

        patchS_stack = torch.cat(patchS_stack)
        patchA_stack = torch.cat(patchA_stack)
        patchB_stack = torch.cat(patchB_stack)
        mixedS = torch.cat(mixedS)
        mixedA = torch.cat(mixedA)
        mixedB = torch.cat(mixedB)

        mixedT = mixedA * mixedS + mixedB
        patchT_stack = patchA_stack * patchS_stack + patchB_stack
        patchSD = torch.norm(mixedS - patchS_stack, 2, 2).exp()
        wgt = patchSD / patchSD.sum(1, keepdim=True)
        term1 = torch.pow(mixedT - patchT_stack, 2).sum(2)
        loss_nl = torch.sum(wgt * term1, 1).mean()

        return loss_nl

    def train(self, total_iter=250):
        optimizer = optim.Adam([self.paramA, self.paramB], lr=0.1, weight_decay=0)
        hyper_l = 0.005
        hyper_nl = 0.5
        for iter in range(total_iter):
            optimizer.zero_grad()

            loss_d = self.loss_d()
            loss_l = self.loss_l()
            loss_nl = self.loss_nl()
            loss = loss_d + hyper_l * loss_l + hyper_nl * loss_nl

            print("Loss_d: {0:.4f}, Loss_l: {1:.4f}, loss_nl: {2:.4f}".format(loss_d.data, loss_l.data, loss_nl.data))
            if (iter + 1) % 10 == 0:
                print("Iteration:", str(iter + 1) + "/" + str(total_iter), "Loss: {0:.4f}".format(loss.data))
            loss.backward()
            optimizer.step()
