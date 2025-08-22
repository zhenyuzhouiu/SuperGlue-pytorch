import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import argparse
from pathlib import Path
import datetime
import os

from tqdm import tqdm
import matplotlib.cm as cm
import torch.multiprocessing
from torch.autograd import Variable

from torch.utils.data import DataLoader
from data.load_data import SPBatchDataset
from models.superglue_pytorch_batch import SuperGlue
from models.utils import (make_matching_plot,
                          read_image_modified)
from torch.utils.tensorboard import SummaryWriter

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')


def save_ckpt(model, checkpoint_dir, device):
    model.eval()
    model.cpu()
    ckpt_model_filename = checkpoint_dir
    torch.save(model.state_dict(), ckpt_model_filename)
    model.to(device)
    model.train()
    return model


def load(model, checkpoint_dir, device, if_train=True):
    weights_dict = torch.load(checkpoint_dir, map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weights_dict, strict=False)
    model.to(device)
    model = model.train() if if_train else model.eval()
    return model


def parser_arguments():
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=240,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.10,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.05,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[152, 200],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions with [w, h], if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization based on OpenCV instead of Matplotlib')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')

    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--eval_pairs_list', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs for evaluation')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')

    parser.add_argument(
        '--eval_input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--eval_output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optional,'
             'visualizations are written')
    parser.add_argument(
        '--eval_step', type=int, default=8,
        help='the step size for visualization')
    parser.add_argument(
        '--learning_rate', type=int, default=0.0001,
        help='Learning rate')
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='batch_size')
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='num_workers')
    parser.add_argument(
        '--train_path', type=str, default='/mnt/Data/superpoint/data/FINGERKNUCKLE/Left/',
        help='Path to the directory of training imgs.')
    parser.add_argument('--checkpoint_dir', type=str,
                        default="./checkpoint", help="the path for saving checkpoint")
    parser.add_argument('--pretrained_dir', type=str,
                        default="./models/weights/superglue_indoor.pth",
                        help="the pretrained model path")
    parser.add_argument('--checkpoint_step', type=int,
                        default=400, help="save the checkpoint for every iteration")
    parser.add_argument(
        '--epoch', type=int, default=2000,
        help='Number of epoches')
    parser.add_argument("--device", type=str, dest="device", default="cuda:1",
                        help="cuda device 0 or 1, or cpu")

    return parser.parse_args()


def train(opt, writer):
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'detection_threshold': opt.keypoint_threshold,
            'max_num_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    device = opt.device

    # load training data
    train_set = SPBatchDataset(opt.train_path, sp_config=config.get('superpoint', {}),
                               image_size=opt.resize, device=device)
    train_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size,
                              num_workers=opt.num_workers, drop_last=True)

    # load trained SuperGlue
    superglue = SuperGlue(config.get('superglue', {})).to(device)
    if opt.pretrained_dir != "":
        superglue = load(model=superglue, checkpoint_dir=opt.pretrained_dir, device=device, if_train=True)
        print("load the pretrained superglue from the path " + opt.pretrained_dir)

    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    mean_loss = []
    superglue.train()
    # start training
    for epoch in range(1, opt.epoch + 1):
        epoch_loss = 0
        loop = tqdm(enumerate(train_loader), desc='Epoch {}/{}'.format(epoch, opt.epoch), total=len(train_loader))
        for i, pred in loop:
            optimizer.zero_grad()
            for k in pred:
                if k != 'file_name' and k != 'image0' and k != 'image1':
                    if type(pred[k]) == torch.Tensor:
                        if k == 'all_matches':
                            pred[k] = pred[k].to(torch.long).to(device)
                        else:
                            pred[k] = pred[k].to(torch.float).to(device)
                    else:
                        if k == 'all_matches':
                            pred[k] = torch.stack(pred[k]).to(torch.long).to(device)
                        else:
                            pred[k] = torch.stack(pred[k]).to(torch.float).to(device)

            # pred["keypoints"] = [b, n_kps, 2]
            # pred["descriptor"]= [b, 256, n_kps]
            # pred["scores"] = [b, n_kps]
            data = superglue(pred)
            pred = {**pred, **data}

            # process loss
            loss = pred['loss']
            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            mean_loss.append(loss)

            writer.add_scalar("lr", scalar_value=optimizer.param_groups[0]['lr'],
                              global_step=(epoch - 1) * len(train_loader) + i)
            writer.add_scalar("iter_loss", scalar_value=loss.item(), global_step=(epoch - 1) * len(train_loader) + i)
            loop.set_postfix({"iter_loss": "{:.6f}".format(loss.item())})

            # for every eval_step images, print progress and visualize the matches
            if (i + 1) % opt.eval_step == 0:
                ### eval ###
                # Visualize the matches.
                superglue.eval()
                image0, image1 = pred['image0'].cpu().numpy()[0, 0, :, :] * 255., pred['image1'].cpu().numpy()[0, 0, :,
                                                                                  :] * 255.
                kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
                matches, conf = pred['matches0'].cpu().detach().numpy()[0].reshape(-1), pred[
                    'matching_scores0'].cpu().detach().numpy()[0].reshape(-1)
                image0 = read_image_modified(image0, opt.resize, opt.resize_float)
                image1 = read_image_modified(image1, opt.resize, opt.resize_float)
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                viz_path = opt.eval_output_dir / '{}_matches.{}'.format(str(i), opt.viz_extension)
                color = cm.jet(mconf)
                stem = pred['file_name'][0]
                text = []
                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, viz_path, stem, stem, opt.show_keypoints,
                    opt.fast_viz, opt.opencv_display, 'Matches')

                writer.add_scalar("mean_loss", scalar_value=torch.mean(torch.stack(mean_loss)).item(),
                                  global_step=((epoch - 1) * len(train_loader) + i) // opt.eval_step)
                mean_loss = []
                superglue.train()

            # process checkpoint for every checkpoint_step images
            if (i + 1) % opt.checkpoint_step == 0:
                model_out_path = os.path.join(opt.checkpoint_dir, "model_epoch_{}.pth".format('last'))
                save_ckpt(model=superglue, checkpoint_dir=model_out_path, device=device)

        # save checkpoint when an epoch finishes, we should save the best ckpt and last ckpt
        best_loss = 1e5
        epoch_loss /= len(train_loader)
        writer.add_scalar("epoch_loss", scalar_value=epoch_loss, global_step=epoch)
        model_out_path = os.path.join(opt.checkpoint_dir, "model_epoch_{}.pth".format('last'))
        save_ckpt(superglue, model_out_path, device=device)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model_out_path = os.path.join(opt.checkpoint_dir, "model_epoch_{}.pth".format('best'))
            save_ckpt(superglue, model_out_path, device)


if __name__ == '__main__':
    opt = parser_arguments()
    # make sure the flags are properly used
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    print(opt)

    # checkpoint directory
    this_datetime = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    opt.checkpoint_dir = os.path.join(
        opt.checkpoint_dir,
        "{}".format(
            this_datetime
        )
    )
    print("[*] Target Checkpoint Path: {}".format(opt.checkpoint_dir))
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    # log directory
    logdir = os.path.join(opt.checkpoint_dir, 'runs')
    print("[*] Target Logdir Path: {}".format(logdir))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # save hyper parameter
    hyper_parameter = os.path.join(opt.checkpoint_dir, 'hyper_parameter.txt')
    with open(hyper_parameter, 'w') as f:
        for key, value in vars(opt).items():
            f.write('%s:%s\n' % (key, value))

    # store viz results
    eval_output_dir = Path(os.path.join(opt.checkpoint_dir, 'dump_match_pairs'))
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write visualization images to',
          'directory \"{}\"'.format(eval_output_dir))
    opt.eval_output_dir = eval_output_dir

    writer = SummaryWriter(log_dir=logdir)
    train(opt, writer=writer)
