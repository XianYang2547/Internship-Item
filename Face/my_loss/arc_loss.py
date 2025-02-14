import torch
import torch.nn.functional as F

'''平时使用的交叉熵CrossEntropyLoss()是log+softmax+nn.NLLloss(),
ArcFace就是将log+softmax替换成了Arc()，在角度上加了一个值，使得特征间的角度更加小
现在需要一个特征提取器:比如desnet、resnet、mobileNetV2等等,它们的输出形状为(N,feature_dim)
将特征输入ArcFace层，得到输出形状(N,cls)'''


class Arc_loss(torch.nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        # x是（N，V）结构，那么W是（V,C结构），V是特征的维度，C是代表类别数
        self.W = torch.nn.Parameter(torch.randn(feature_num, cls_num))

    def forward(self, x, s=5, m=0.2):
        # "先将特征向量L2归一化，权重L2归一化，他俩的夹角为θ，"
        # "计算cos(θj)，求反余弦arccos(θyi)得到特征xi与真实权值Wyi之间的夹角θyi"
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.W, dim=0)
        # 对cos的结果还要除10，是因为torch.matmul(x,w)的范围不确定，可能会超过1，
        # 这样就超过arccos的定义域范围了，就会产生NaN的结果。当然后续也不需要乘回来，因为w是一个可学习参数，它会自己去改变。学的太慢了，等不了
        cos = torch.matmul(x_norm, w_norm)  # x*w
        a = torch.acos(cos)  # 添加角度间隔m，再求余弦cos(θyj+m)
        top = torch.exp(s * torch.cos(a + m))
        #                   第一项(N,1)  keepdim=True保持形状不变.这是我们原有的softmax的分布。第二项(N,C),最后结果是(N,C)
        down2 = torch.sum(torch.exp(s * torch.cos(a)), dim=1, keepdim=True) - torch.exp(s * torch.cos(a))
        outs = torch.log(top / (top + down2))
        return outs


if __name__ == '__main__':
    arc = Arc_loss(10, 20)
    data = torch.randn(1, 10)
    out = arc(data)
    print(out)
    print(torch.sum(out))
