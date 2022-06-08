import torch


# TODO: update this, split into 3 functions for corresponding targets, remove 'feature_arrays'
def gen_trx_data(lengths, target_share=.5, target_type='bin_cls'):
    n = len(lengths)
    if target_type == 'bin_cls':
        targets = (torch.rand(n) >= target_share).long()
    elif target_type == 'multi_cls':
        targets = torch.randint(0, 4, (n,))
    elif target_type == 'regression':
        targets = torch.randn(n)
    else:
        raise AttributeError(f'Unknown target_type: {target_type}')

    samples = list()
    for target, length in zip(targets, lengths):
        s = dict()
        s['trans_type'] = (torch.rand(length)*10 + 1).long()
        s['mcc_code'] = (torch.rand(length) * 20 + 1).long()
        s['amount'] = (torch.rand(length) * 1000 + 1).long()

        samples.append({'feature_arrays': s, 'target': target})

    return samples
