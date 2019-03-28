import numpy as np
import torch


class PatchMatch:
    def __init__(self, a, b, patch_size=3):
        self.a = a
        self.b = b
        self.ah = a.shape[0]
        self.aw = a.shape[1]
        self.bh = b.shape[0]
        self.bw = b.shape[1]
        self.patch_size = patch_size

        self.nnf = np.zeros((self.ah, self.aw, 2)).astype(np.int)  # The NNF
        self.nnd = np.zeros((self.ah, self.aw))  # The NNF distance map
        self.init_nnf()

    def init_nnf(self):
        for ay in range(self.ah):
            for ax in range(self.aw):
                by = np.random.randint(self.bh)
                bx = np.random.randint(self.bw)
                self.nnf[ay, ax] = [by, bx]
                self.nnd[ay, ax] = self.calc_dist(ay, ax, by, bx)

    def calc_dist(self, ay, ax, by, bx):
        """
            Measure distance between 2 patches across all channels
            ay : y coordinate of a patch in a
            ax : x coordinate of a patch in a
            by : y coordinate of a patch in b
            bx : x coordinate of a patch in b
        """
        dy0 = dx0 = self.patch_size // 2
        dy1 = dx1 = self.patch_size // 2 + 1
        dy0 = min(ay, by, dy0)
        dy1 = min(self.ah - ay, self.bh - by, dy1)
        dx0 = min(ax, bx, dx0)
        dx1 = min(self.aw - ax, self.bw - bx, dx1)

        dist = np.sum(np.square(self.a[ay - dy0:ay + dy1, ax - dx0:ax + dx1] - self.b[by - dy0:by + dy1, bx - dx0:bx + dx1]))
        dist /= ((dy0 + dy1) * (dx0 + dx1))
        return dist

    def improve_guess(self, ay, ax, by, bx, ybest, xbest, dbest):
        d = self.calc_dist(ay, ax, by, bx)
        if d < dbest:
            ybest, xbest, dbest = by, bx, d
        return ybest, xbest, dbest

    def improve_nnf(self, total_iter=5):
        for iter in range(total_iter):
            if iter % 2:
                ystart, yend, ychange = self.ah - 1, -1, -1
                xstart, xend, xchange = self.aw - 1, -1, -1
            else:
                ystart, yend, ychange = 0, self.ah, 1
                xstart, xend, xchange = 0, self.aw, 1

            for ay in range(ystart, yend, ychange):
                for ax in range(xstart, xend, xchange):
                    ybest, xbest = self.nnf[ay, ax]
                    dbest = self.nnd[ay, ax]

                    # Propagation
                    if 0 <= (ay - ychange) < self.ah:
                        yp, xp = self.nnf[ay - ychange, ax]
                        yp += ychange
                        if 0 <= yp < self.bh:
                            ybest, xbest, dbest = self.improve_guess(ay, ax, yp, xp, ybest, xbest, dbest)
                    if 0 <= (ax - xchange) < self.aw:
                        yp, xp = self.nnf[ay, ax - xchange]
                        xp += xchange
                        if 0 <= xp < self.bw:
                            ybest, xbest, dbest = self.improve_guess(ay, ax, yp, xp, ybest, xbest, dbest)

                    # Random search
                    rand_d = max(self.bh, self.bw)
                    while rand_d >= 1:
                        ymin, ymax = max(ybest - rand_d, 0), min(ybest + rand_d, self.bh)
                        xmin, xmax = max(xbest - rand_d, 0), min(xbest + rand_d, self.bw)
                        yp = np.random.randint(ymin, ymax)
                        xp = np.random.randint(xmin, xmax)
                        ybest, xbest, dbest = self.improve_guess(ay, ax, yp, xp, ybest, xbest, dbest)
                        rand_d = rand_d // 2

                    self.nnf[ay, ax] = [ybest, xbest]
                    self.nnd[ay, ax] = dbest
            print("iteration:", str(iter + 1) + "/" + str(total_iter))

    def solve(self):
        self.improve_nnf(total_iter=8)


def bds_vote(ref, nnf_sr, nnf_rs, patch_size=3):
    """
    Reconstructs an image or feature map by bidirectionaly
    similarity voting
    """

    src_height = nnf_sr.shape[0]
    src_width = nnf_sr.shape[1]
    ref_height = nnf_rs.shape[0]
    ref_width = nnf_rs.shape[1]
    channel = ref.shape[0]

    guide = np.zeros((channel, src_height, src_width))
    weight = np.zeros((src_height, src_width))
    ws = 1 / (src_height * src_width)
    wr = 1 / (ref_height * ref_width)

    # coherence
    # The S->R forward NNF enforces coherence
    for sy in range(src_height):
        for sx in range(src_width):
            ry, rx = nnf_sr[sy, sx]

            dy0 = dx0 = patch_size // 2
            dy1 = dx1 = patch_size // 2 + 1
            dy0 = min(sy, ry, dy0)
            dy1 = min(src_height - sy, ref_height - ry, dy1)
            dx0 = min(sx, rx, dx0)
            dx1 = min(src_width - sx, ref_width - rx, dx1)

            guide[:, sy - dy0:sy + dy1, sx - dx0:sx + dx1] += ws * ref[:, ry - dy0:ry + dy1, rx - dx0:rx + dx1]
            weight[sy - dy0:sy + dy1, sx - dx0:sx + dx1] += ws

    # completeness
    # The R->S backward NNF enforces completeness
    for ry in range(ref_height):
        for rx in range(ref_width):
            sy, sx = nnf_rs[ry, rx]

            dy0 = dx0 = patch_size // 2
            dy1 = dx1 = patch_size // 2 + 1
            dy0 = min(sy, ry, dy0)
            dy1 = min(src_height - sy, ref_height - ry, dy1)
            dx0 = min(sx, rx, dx0)
            dx1 = min(src_width - sx, ref_width - rx, dx1)

            guide[:, sy - dy0:sy + dy1, sx - dx0:sx + dx1] += wr * ref[:, ry - dy0:ry + dy1, rx - dx0:rx + dx1]
            weight[sy - dy0:sy + dy1, sx - dx0:sx + dx1] += wr

    weight[weight == 0] = 1
    guide /= weight
    return guide
