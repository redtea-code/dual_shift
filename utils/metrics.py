import torch

eps = 1e-8


def pairwise_distances(x):
    # xshouldbetwodimensional
    if x.dim() == 1:
        x = x.unsqueeze(1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_sigma(Z, dim=2, p=2):
    dist_matrix = torch.norm(Z[:, None] - Z, dim, p)
    sigma = torch.mean(torch.sort(dist_matrix[:, :10], 1)[0])
    if sigma < 0.1:
        sigma = 0.1
    return sigma


def calculate_sigma_label(Z, dim=2, p=2):
    if Z.dim() == 1:
        Z = Z.unsqueeze(1)
    dist_matrix = torch.norm(Z[:, None] - Z, dim, p)
    sigma = torch.mean(torch.sort(dist_matrix[:, :10], 1)[0])
    if sigma < 0.1:
        sigma = 0.1
    return sigma


def calculate_gram_mat(x, sigma):
    # x = torch.mean(x, dim=0)
    dist = pairwise_distances(x)
    # dist = dist/torch.max(dist)
    # a=torch.exp(-dist /sigma)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.01
    k = calculate_gram_mat(x, sigma)
    k = k / (torch.trace(k) + eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / (torch.trace(k) + eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy_label(x, y, s_x, s_y):
    if x.dim() == 1:
        x = x.unsqueeze(1)
    alpha = 1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / (torch.trace(k) + eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy3(x, y, z, s_x, s_y, s_z):
    if y.dim() == 1:
        y = y.unsqueeze(1)
    alpha = 1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    z = calculate_gram_mat(z, s_z)
    k = torch.mul(x, y)
    k = torch.mul(k, z)
    k = k / (torch.trace(k) + eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def calculate_conditional_MI(x, y, z):
    #计算条件互信息
    s_x = calculate_sigma(x) ** 2
    s_y = calculate_sigma_label(y) ** 2
    s_z = calculate_sigma(z) ** 2
    Hyz = joint_entropy_label(y, z, s_y, s_z)
    Hxz = joint_entropy(x, z, s_x, s_z)
    Hz = reyi_entropy(z, sigma=s_z)
    Hxyz = joint_entropy3(x, y, z, s_x, s_y, s_z)
    CI = Hyz + Hxz - Hz - Hxyz
    return CI


def calculate_MI(x, y):
    #计算互信息
    s_x = calculate_sigma(x)
    s_y = calculate_sigma(y)
    Hx = reyi_entropy(x, s_x ** 2)
    Hy = reyi_entropy(y, s_y ** 2)
    Hxy = joint_entropy(x, y, s_x ** 2, s_y ** 2)
    Ixy = Hx + Hy - Hxy
    return Ixy
