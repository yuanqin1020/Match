import torch


def euclidean_distance(x, y):
    return torch.sum((x - y) ** 2, dim=-1)


def approximate_hamming_similarity(x, y):
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)


def pairwise_loss(x, y, labels, loss_type='margin', margin=1.0):

    labels = labels.float()

    if loss_type == 'margin':
        return torch.relu(margin - labels * (1 - euclidean_distance(x, y)))
    elif loss_type == 'hamming':
        return 0.25 * (labels - approximate_hamming_similarity(x, y)) ** 2
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)


def triplet_loss(x_1, y, x_2, z, loss_type='margin', margin=1.0):
    if loss_type == 'margin':
        return torch.relu(margin +
                          euclidean_distance(x_1, y) -
                          euclidean_distance(x_2, z))
    elif loss_type == 'hamming':
        return 0.125 * ((approximate_hamming_similarity(x_1, y) - 1) ** 2 +
                        (approximate_hamming_similarity(x_2, z) + 1) ** 2)
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)

def compute_cross_attention(x, y, sim):

    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y
