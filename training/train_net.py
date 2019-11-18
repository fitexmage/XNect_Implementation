import torch
import os

from tensorboardX import SummaryWriter
from tqdm import tqdm
from visualization.visualize import visualize_output


def step(data_loader, model, criterion_hm, criterion_paf, to_train=False, optimizer=None, viz_output=False, epoch=0, writer=None):
    if to_train:
        model.train()
    else:
        model.eval()
    nIters = len(data_loader)
    hm_loss_meter, paf_loss_meter = AverageMeter(), AverageMeter()
    with tqdm(total=nIters) as t:
        for i, (input_, heatmap, paf, ignore_mask, indices) in enumerate(data_loader):
            input_cuda = input_.float().cuda()
            heatmap_t_cuda = heatmap.float().cuda()
            paf_t_cuda = paf.float().cuda()
            ignore_mask_cuda = ignore_mask.reshape(ignore_mask.shape[0], 1,
                                                   ignore_mask.shape[1], ignore_mask.shape[2]).float().cuda()

            allow_mask = 1 - ignore_mask_cuda
            heatmap_outputs, paf_outputs = model(input_cuda)
            loss_hm_total = 0
            loss_paf_total = 0
            # for i in range(len(heatmap_outputs)):
            #     heatmap_out = heatmap_outputs[i]
            #     paf_out = paf_outputs[i]
            #     loss_hm_total += criterion_hm(heatmap_out * allow_mask, heatmap_t_cuda * allow_mask)/allow_mask.sum().detach()/heatmap.shape[0]/heatmap.shape[1]
            #     loss_paf_total += criterion_paf(paf_out * allow_mask, paf_t_cuda * allow_mask)/allow_mask.sum().detach()/heatmap.shape[0]/paf.shape[1]
            loss_hm_total += criterion_hm(heatmap_outputs * allow_mask, heatmap_t_cuda * allow_mask)/allow_mask.sum().detach()
            loss_paf_total += criterion_paf(paf_outputs * allow_mask, paf_t_cuda * allow_mask)/allow_mask.sum().detach()

            loss = loss_hm_total + loss_paf_total
            output = (heatmap_outputs[-1].data.cpu().numpy(), paf_outputs[-1].data.cpu().numpy(), indices.numpy())
            if to_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if viz_output:
                visualize_output(input_.numpy(), heatmap.numpy(), paf.numpy(), ignore_mask.numpy(), output)
            hm_loss_meter.update(loss_hm_total.data.cpu().numpy())
            paf_loss_meter.update(loss_paf_total.data.cpu().numpy())
            t.set_postfix(loss_hm='{:05.3f}'.format(hm_loss_meter.avg), loss_paf='{:05.3f}'.format(paf_loss_meter.avg))
            t.update()
            if to_train:
                writer.add_scalar('hm loss', hm_loss_meter.avg, global_step=nIters * (epoch - 1) + i)
                writer.add_scalar('paf loss', paf_loss_meter.avg, global_step=nIters * (epoch - 1) + i)
    return hm_loss_meter.avg, paf_loss_meter.avg


def train_net(train_loader, test_loader, model, criterion_hm, criterion_paf, optimizer,
              n_epochs, val_interval, learn_rate, drop_lr, save_dir, viz_output=False, latest_inx=0):
    heatmap_loss_avg, paf_loss_avg = 0.0, 0.0
    for epoch in range(latest_inx + 1, n_epochs + 1):
        writer = SummaryWriter(os.path.join(save_dir, 'runs'))
        print("Epoch: ", epoch)
        adjust_learning_rate(optimizer, epoch, drop_lr, learn_rate)
        heatmap_loss_avg, paf_loss_avg = step(train_loader, model, criterion_hm, criterion_paf, True, optimizer, viz_output=viz_output, epoch=epoch, writer=writer)
        print("Training Heatmap Loss: ", heatmap_loss_avg)
        print("Training PAF Loss: ", paf_loss_avg)
        if epoch % val_interval == 0:
            heatmap_loss_avg, paf_loss_avg = validate_net(test_loader, model, criterion_hm, criterion_paf, save_dir, epoch, viz_output=viz_output)
            print("Validation Heatmap Loss: ", heatmap_loss_avg)
            print("Validation PAF Loss: ", paf_loss_avg)
            print()
            writer.add_scalar('validation hm loss', heatmap_loss_avg, global_step=epoch)
            writer.add_scalar('validation paf loss', paf_loss_avg, global_step=epoch)
    return heatmap_loss_avg, paf_loss_avg


def validate_net(test_loader, model, criterion_hm, criterion_paf, save_dir=None, epoch=0, viz_output=False):
    heatmap_loss_avg, paf_loss_avg = step(test_loader, model, criterion_hm, criterion_paf, False, viz_output=viz_output)
    if not save_dir is None:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model', 'model_{}.pth'.format(epoch)))
    return heatmap_loss_avg, paf_loss_avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    lr = LR * (0.25 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Learning Rate:", lr)
