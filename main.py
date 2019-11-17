import numpy as np
import torch
import random
from opts.base_opts import Opts
from data_process.data_loader_provider import create_data_loaders
from model.model_provider import create_model, create_optimizer
from training.train_net import train_net, validate_net
from evaluation.eval_net import eval_net

def main():
    # Seed all sources of randomness to 0 for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    opt = Opts().parse()

    # Create data loaders
    train_loader, test_loader = create_data_loaders(opt)

    # Create nn
    model, criterion_hm, criterion_paf, latest_inx = create_model(opt)
    model = model.cuda()
    criterion_hm = criterion_hm.cuda()
    criterion_paf = criterion_paf.cuda()

    # Create optimizer
    optimizer = create_optimizer(opt, model)

    # Other params
    n_epochs = opt.nEpoch
    to_train = opt.train
    drop_lr = opt.dropLR
    val_interval = opt.valInterval
    learn_rate = opt.LR
    visualize_out = opt.vizOut

    # train/ test
    # train_net(train_loader, test_loader, model, criterion_hm, criterion_paf, optimizer, n_epochs,
    #           val_interval, learn_rate, drop_lr, opt.saveDir, visualize_out, latest_inx)

    model.eval()
    dataset = test_loader.dataset
    scales = [1., 0.5, 0.75, 1.25, 1.5, 2.0]
    assert (scales[0] == 1)
    n_scales = len(scales)
    dataset_len = 100  # len(dataset)

    with torch.no_grad():
        for i in range(dataset_len):
            imgs, heatmap_t, paf_t, ignore_mask_t, keypoints = dataset.get_imgs_multiscale(i, scales,flip=False)
            n_imgs = len(imgs)
            assert (n_imgs == n_scales)
            heights = list(map(lambda x: x.shape[1], imgs))
            widths = list(map(lambda x: x.shape[2], imgs))
            max_h, max_w = max(heights), max(widths)
            imgs_np = np.zeros((n_imgs, 3, max_h, max_w))
            for j in range(n_imgs):
                img = imgs[j]
                h, w = img.shape[1], img.shape[2]
                imgs_np[j, :, :h, :w] = img
            img_basic = imgs[0]
            heatmap_avg_lst = []
            paf_avg_lst = []

            from data_process.process_utils import resize_hm
            for j in range(0, n_imgs):
                imgs_torch = torch.from_numpy(imgs_np[j:j + 1]).float().cuda()
                heatmaps, pafs = model(imgs_torch)
                heatmap = heatmaps.data.cpu().numpy()[0, :, :heights[j] // 8, :widths[j] // 8]
                paf = pafs.data.cpu().numpy()[0, :, :heights[j] // 8, :widths[j] // 8]
                heatmap = resize_hm(heatmap, (widths[0], heights[0]))
                paf = resize_hm(paf, (widths[0], heights[0]))
                heatmap_avg_lst += [heatmap]
                paf_avg_lst += [paf]
            heatmap_avg = sum(heatmap_avg_lst) / n_imgs
            paf_avg = sum(paf_avg_lst) / n_imgs
            print(np.transpose(img_basic, (1, 2, 0)).shape)
            print(heatmap_avg.shape)
            print(paf_avg.shape)

    # validate_net(test_loader, model, criterion_hm, criterion_paf, viz_output=visualize_out)

if __name__ == '__main__':
    main()
