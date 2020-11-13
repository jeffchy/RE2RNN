import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def val(model, intent_dataloader, epoch, mode='DEV', logger=None, config=None, i2in=None, criterion=torch.nn.CrossEntropyLoss()):
    assert mode in ['TRAIN', 'DEV', 'TEST', 'INIT']

    avg_loss = 0
    acc = 0

    if intent_dataloader is None:
        return None, None, None, None, None

    model.eval()
    pbar_dev = tqdm(intent_dataloader)
    pbar_dev.set_description("VAL {}".format(mode))
    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch in pbar_dev:

            if config.model_type == 'Onehot':
                x = batch['x']
            else:
                x_forward = batch['x_forward']
                x_backward = batch['x_backward']

            label = batch['i'].view(-1)
            lengths = batch['l']

            if torch.cuda.is_available():
                if config.model_type == 'Onehot':
                    x = x.cuda()
                else:
                    x_forward = x_forward.cuda()
                    x_backward = x_backward.cuda()

                lengths = lengths.cuda()
                label = label.cuda()

            if config.bidirection:
                out = model(x_forward, x_backward, lengths)
            else:
                if config.model_type == 'Onehot':
                    out = model(x, lengths)
                else:
                    out = model(x_forward, lengths)

            loss = criterion(out, label)
            avg_loss += loss.item()
            acc += (out.argmax(1) == label).sum().item()
            all_pred += list(out.argmax(1).cpu().numpy())
            all_label += list(label.cpu().numpy())

            pbar_dev.set_postfix_str("{} - total right: {}, total entropy loss: {}".format(mode, acc, loss))

    acc = acc / len(intent_dataloader.dataset)
    avg_loss = avg_loss / len(intent_dataloader.dataset)
    p_micro, r_micro = acc, acc
    if 'SMS' in config.dataset:
        p_micro = precision_score(all_label, all_pred, average='binary', pos_label=1)
        r_micro = recall_score(all_label, all_pred, average='binary', pos_label=1)

    print('Dataset Len: {}'.format(len(intent_dataloader.dataset)))
    print("{} Epoch: {} | ACC: {}, LOSS: {}, P: {}, R: {}".format(mode, epoch, acc, avg_loss, p_micro, r_micro))
    if logger:
        logger.add("{} Epoch: {} | ACC: {}, LOSS: {}, P: {}, R: {}".format(mode, epoch, acc, avg_loss,  p_micro, r_micro))

    if config.only_probe:
        confusion_mat = confusion_matrix(all_label, all_pred, labels=[i for i in range(len(i2in))])
        labels = [i2in[i] for i in range(len(i2in))]
        fig = plt.figure()
        fig.set_size_inches(18, 18)
        cmap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=False)
        g = sns.heatmap(confusion_mat, annot=True,   cmap=cmap, linewidths=1,
                        linecolor='gray', xticklabels=labels, yticklabels=labels,)
        plt.show()

    return acc, avg_loss, p_micro, r_micro


def val_marry(model, intent_dataloader, epoch, mode='DEV', config=None, logger=None, criterion=torch.nn.CrossEntropyLoss()):
    # assert mode in ['DEV', 'TEST']

    avg_loss = 0
    acc = 0

    if not intent_dataloader:
        return None, None, None, None

    model.eval()
    pbar_dev = tqdm(intent_dataloader)
    pbar_dev.set_description("VAL {}".format(mode))
    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch in pbar_dev:

            x = batch['x']
            label = batch['i'].view(-1)
            lengths = batch['l']
            re_tag = batch['re']

            if torch.cuda.is_available():
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()
                re_tag = re_tag.cuda()

            out = model(x, lengths, re_tag)
            loss = criterion(out, label)
            avg_loss += loss.item()
            acc += (out.argmax(1) == label).sum().item()

            all_pred += list(out.argmax(1).cpu().numpy())
            all_label += list(label.cpu().numpy())

            pbar_dev.set_postfix_str("{} - total right: {}, total entropy loss: {}".format(mode, acc, loss))

    acc = acc / len(intent_dataloader.dataset)
    avg_loss = avg_loss / len(intent_dataloader.dataset)
    p_micro, r_micro = acc, acc
    if config.dataset == 'SMS':
        p_micro = precision_score(all_label, all_pred, average='binary', pos_label=1)
        r_micro = recall_score(all_label, all_pred, average='binary', pos_label=1)

    print("{} Epoch: {} | ACC: {}, LOSS: {}, P: {}, R: {}".format(mode, epoch, acc, avg_loss, p_micro, r_micro))
    if logger:
        logger.add("{} Epoch: {} | ACC: {}, LOSS: {}, P: {}, R: {}".format(mode, epoch, acc, avg_loss,  p_micro, r_micro))

    return acc, avg_loss, p_micro, r_micro
