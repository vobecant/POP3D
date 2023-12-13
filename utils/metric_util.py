import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb
import time
try:
    from mmseg.utils import get_root_logger
except:
    from mmengine.logging import MMLogger


def get_confusion_matrix(gt_label, pred_label, class_num, cluster_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :param cluster: the number of clusters, aka pseudo-classes.
    :return: the confusion matrix
    """
    index = (gt_label * cluster_num + pred_label).astype('int32')
    label_count = np.bincount(index, minlength=class_num * cluster_num)
    confusion_matrix = label_count.reshape(class_num, cluster_num).astype('float32')
    return confusion_matrix

def get_confusion_matrix_torch(gt_label, pred_label, class_num, cluster_num):
    index = (gt_label * cluster_num + pred_label).long()
    label_count = torch.bincount(index, minlength=class_num * cluster_num)
    confusion_matrix = label_count.reshape(class_num, cluster_num).float()
    return confusion_matrix


def roc(predictions, targets, n_steps=1000):
    '''

    Args:
        predictions: similarities of the retrieval, higher is better
        targets: 1/0 binary targets
        n_steps: We evaluate the ROC curve at this many steps

    Returns:

    '''
    sorted_idx = torch.argsort(predictions, descending=True)
    targets_sorted = targets[sorted_idx]
    P = targets_sorted.sum()  # positives
    N = len(targets_sorted) - P  # negatives

    n_predictions = len(predictions)  # number of predictions
    step_size = n_predictions / n_steps  # We evaluate after every step of this size.

    tprs, fprs = [], []

    step_end = step_size

    while step_end < n_predictions:
        step_end = min([step_end, n_predictions])  # until which point should we compute now
        step_end = int(step_end)
        tgt_samples = targets_sorted[:step_end]
        tp = tgt_samples.sum().float()  # sum of 1
        fp = len(tgt_samples) - tp  # sum of 0

        tpr = tp / P
        tprs.append(tpr)
        fpr = fp / N
        fprs.append(fpr)

        step_end += step_size

    plt.plot(fprs, tprs)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC curve')
    plt.show()

    return tprs, fprs


class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name,
                 sub=None,
                 extra_classes=None,
                 extra_classes_pred=None
                 # empty_class: int
                 ):
        self.class_indices = class_indices
        self.class_indices_torch = torch.tensor(self.class_indices).cuda()
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name
        self.sub = sub
        self.extra_classes = extra_classes
        self.extra_classes_pred = extra_classes_pred if extra_classes_pred is not None else 0

        if self.num_classes > 2:
            if self.extra_classes is not None:
                self.confusion_matrix = torch.zeros(
                    (self.num_classes + self.extra_classes,
                     self.num_classes + self.extra_classes + self.extra_classes_pred)
                )  # GT vs predicted
            else:
                self.confusion_matrix = torch.zeros((self.num_classes + 1, self.num_classes + 1))  # GT vs predicted
        else:
            self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))  # GT vs predicted

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def get_logger(self):
        try:
            logger = get_root_logger(log_level='INFO')
        except:
            logger = MMLogger.get_current_instance()
        return logger

    def _after_step(self, outputs, targets, return_cur_miou=False, rank=None):
        logger = self.get_logger()

        t1s=time.time()
        targets_nonignore = targets != self.ignore_label
        outputs = outputs[targets_nonignore]
        targets = targets[targets_nonignore]
        if return_cur_miou:
            out_str = '; '.join([f'{n}: {c}' for n, c in zip(*outputs.unique(return_counts=True))])
            tgt_str = '; '.join([f'{n}: {c}' for n, c in zip(*targets.unique(return_counts=True))])
            print(f'outputs: {out_str}\ntargets: {tgt_str}')
        t1=time.time()-t1s


        t3s=time.time()
        # for i, c in enumerate(self.class_indices):
        #     total_seen_cur[i] = torch.sum(targets == c)#.item()
        #     self.total_seen[i] += total_seen_cur[i]
        #     total_correct_cur[i] = torch.sum((targets == c) & (outputs == c))#.item()
        #     self.total_correct[i] += total_correct_cur[i]
        #     total_positive_cur[i] = torch.sum(outputs == c)#.item()
        #     self.total_positive[i] += total_positive_cur[i]
        # t3=time.time()-t3s

        t_c = (targets[:,None] == self.class_indices_torch.unsqueeze(0))
        o_c = (outputs[:,None] == self.class_indices_torch.unsqueeze(0))
        t_and_o = t_c & o_c

        total_seen_cur = t_c.sum(0)
        self.total_seen += total_seen_cur
        total_correct_cur = t_and_o.sum(0)
        self.total_correct += total_correct_cur
        total_positive_cur = o_c.sum(0)
        self.total_positive += total_positive_cur
        t3=time.time()-t3s

        if return_cur_miou:
            ious = []
            for i in range(self.num_classes):
                if total_seen_cur[i] == 0:
                    # ious.append(1)
                    if self.total_positive[i] > 0:
                        ious.append(0)
                else:
                    cur_iou = total_correct_cur[i] / (total_seen_cur[i] + total_positive_cur[i] - total_correct_cur[i])
                    ious.append(cur_iou.item())
            miou = np.mean(ious) * 100

        t4s=time.time()
        sub = self.sub if self.sub is not None else 0 if self.num_classes == 2 else 1
        extra_classes = sub if self.extra_classes is None else self.extra_classes
        class_num = self.num_classes + extra_classes
        cluster_num = self.num_classes + extra_classes + self.extra_classes_pred
        t4=time.time()-t4s

        t5s=time.time()
        try:
            self.confusion_matrix += get_confusion_matrix(
                targets.cpu().numpy().astype(np.uint8) - sub, outputs.cpu().numpy().astype(np.uint8) - sub, class_num,
                cluster_num)
            # self.confusion_matrix += get_confusion_matrix_torch(targets - sub, outputs - sub, class_num, cluster_num)
        except:
            print(f'name: {self.name}, targets: {targets.unique()}, outputs: {outputs.unique()}, '
                  f'class_num: {class_num}, cluster_num: {cluster_num}')
        t5=time.time()-t5s

        rank = torch.distributed.get_rank() if rank is None else rank
        time_total = time.time()-t1s
        # to_print = '[rank {}, {}] t1={:.1f}s, t3={:.1f}s, t4={:.1f}s, t5={:.1f}s, total={:.1f}s'.format(rank,self.name,t1,t3,t4,t5,time_total)
        # logger.info(to_print)
        # print(to_print)


        if return_cur_miou:
            return miou

    def _after_epoch(self, log_wandb=False, tag=None, step=None, return_per_class=False):
        try:
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
        except:
            pass

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        if self.num_classes == 2:
            # Per ground-truth class true positives.
            tp = torch.diag(self.confusion_matrix)
            # False positives = predict a GT label somewhere, where it actually isn't.
            fp = torch.sum(self.confusion_matrix, axis=1) - tp
            # False negative = Missed GT labels.
            fn = torch.sum(self.confusion_matrix, axis=0) - tp

            precision = torch.nanmean(tp / (tp + fp)) * 100
            recall = torch.nanmean(tp / (tp + fn)) * 100

        confusion_matrix_clipped = torch.clip(self.confusion_matrix, 1)
        pos = confusion_matrix_clipped.sum(1)  # sums each row -> has as many elements as there are rows
        res = confusion_matrix_clipped.sum(0)

        pos_expanded = pos.unsqueeze(-1).expand(self.confusion_matrix.shape[0], self.confusion_matrix.shape[1])
        res_expanded = res.unsqueeze(0).expand(self.confusion_matrix.shape[0], self.confusion_matrix.shape[1])

        miou = np.mean(ious)
        logger = self.get_logger()
        logger.info(f'\nValidation per class iou {self.name}:')
        for iou, label_str in zip(ious, self.label_str):
            logger.info('\t%s : %.2f%%' % (label_str, iou * 100))
            if log_wandb:
                wandb.log({f"{tag}_{label_str}": iou * 100},
                          # step=step,
                          commit=False)
        if self.num_classes == 2:
            logger.info('\tprecision: {:.2f}'.format(precision))
            logger.info('\trecall: {:.2f}'.format(recall))
        print()

        print(f'\nValidation per class iou {self.name}:')
        for iou, label_str in zip(ious, self.label_str):
            print('\t%s : %.2f%%' % (label_str, iou * 100))
            if log_wandb:
                wandb.log({f"{tag}_{label_str}": iou * 100},
                          # step=step,
                          commit=False)
        if self.num_classes == 2:
            print('\tprecision: {:.2f}'.format(precision))
            print('\trecall: {:.2f}'.format(recall))
        print()

        confusion_matrix_rownorm = confusion_matrix_clipped / pos_expanded
        confusion_matrix_colnorm = confusion_matrix_clipped / res_expanded
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(confusion_matrix_rownorm)
        ax[0].set_title('row-normalized')
        ax[1].imshow(confusion_matrix_colnorm)
        ax[1].set_title('col-normalized')
        plt.suptitle(f'{self.name}, confusion matrix, miou={miou}')
        plt.show()

        if return_per_class:
            return miou * 100, [iou * 100 for iou in ious]
        return miou * 100
