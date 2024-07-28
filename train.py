import argparse
import random
import torch
import numpy as np
import RL_base.UTILS as utils
import albumentations as A
from RL_base.model import policy_network
from data.dataset import Compose, FSSDataset
from RL_base.UTILS import prepare_save_vit_h5py, prepare_cluster_h5py
from RL_base.learn_policy import policy_gradient_train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--cluster_num',
                        type=int,
                        default=10,
                        help='Chose cluster number')
    parser.add_argument('--limit_num',
                        type=int,
                        default=10,
                        help='the minimum num of a cluster')
    parser.add_argument('--sample_num',
                        type=int,
                        default=2,
                        help='Sample prompt number')
    parser.add_argument('--model_name',
                        type=str,
                        default='vit_large_patch14_clip_224.laion2b_ft_in12k_in1k',
                        help='Policy network Pre-training model.')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default='./pretrained_model/pytorch_model.bin',
                        help='Policy network Pre-training model checkpoint path.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--base_dir', default='./datasets',
                        help='pascal base dir')
    parser.add_argument('--fold', default=0, type=int)
    # parser.add_argument('--split', default='val', type=str)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--percentage', default='', type=str)
    parser.add_argument('--offline_cluster',
                        default=True,
                        type=bool,
                        help='offline cluster before inference or train')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--task', default='segmentation', choices=['segmentation', 'detection'])
    parser.add_argument('--dataset_type', default='pascal',
                        choices=['pascal', 'pascal_det'])
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--BENCHMARK', default='pascal', type=str, help='dataset type:"pascal, fss, coco, iSALD"')
    parser.add_argument('--cats_augmentation', default=True, type=bool)
    parser.add_argument('--pfenet_augmentation', default=False, type=bool)
    parser.add_argument('--candidate_num', default=50, type=int, help='the max candidate examples num')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--output_root', type=str, default='log')
    parser.add_argument('--engine_ckpt_path', type=str, help='path to ckpt',
                        default='./pretrained_model/seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types',
                        choices=['instance', 'semantic'], default='semantic')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                        help='tau(temperature) (default: 1.0)')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='hard Gumbel softmax')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--resume', type=bool, default=False,
                        help='Breakpoint recovery training')
    parser.add_argument('--backbone', default="VIT", type=str, help='backbone of retrival feature')
    parser.add_argument('--method', type=str, default='clst', help='random, clst')
    parser.add_argument('--process_img_size', default=224, type=int, help='input size of VIT')
    parser.add_argument('--Tem', default=1, type=int, help='Temperature')
    parser.add_argument('--mode', default='train', type=str, help='train, val')
    args = parser.parse_args()


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.output_path = os.path.join(args.output_root, args.BENCHMARK)
    utils.create_dir(args.output_path)
    return args


if __name__ == '__main__':

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU

    ## policy network
    policy_model = policy_network(model_name=args.model_name,
                                  add_linear=True,
                                  embedding_size=args.embedding_size,
                                  checkpoint_path=args.ckpt_path,
                                  freeze_encoder=True)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    policy_model = policy_model.to(device)

    # Dataset initialization
    train_Img_size = args.process_img_size
    datapath = os.path.join(args.base_dir, 'pascal-5i')

    FSSDataset.initialize(benchmark=args.BENCHMARK, img_size=args.process_img_size,
                          datapath=datapath, use_original_imgsize=False,
                          apply_cats_augmentation=args.cats_augmentation,
                          apply_pfenet_augmentation=args.pfenet_augmentation)
    dataloader_trn = FSSDataset.build_dataloader(args.BENCHMARK, args.batch_size,
                                                 args.num_workers, args.fold, 'trn', shot=1)


    ### save feature
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    img_metadata_classwise_trn = dataloader_trn.sampler.data_source.img_metadata_classwise
    can_supports = {}

    for class_sample in range(len(img_metadata_classwise_trn)):
        class_sample_length = len(img_metadata_classwise_trn[class_sample])
        if class_sample_length == 0:
            continue
        can_supports[class_sample] = img_metadata_classwise_trn[class_sample]


    transform = Compose([
        A.Resize(train_Img_size, train_Img_size),
        A.Normalize(img_mean, img_std),
        A.pytorch.transforms.ToTensorV2(),
    ])

    VIT_name = os.path.join(args.output_path, args.BENCHMARK + "-train_VIT_v2.h5")

    # dataset preprocess
    if os.path.exists(VIT_name):
        pass
    else:
        prepare_save_vit_h5py(VIT_name, can_supports, datapath, policy_model, transform, device, args)
    if args.offline_cluster:
        prepare_cluster_h5py(VIT_name, can_supports.keys(), can_supports, device, args)

    # agent trained
    policy_gradient_train(policy_model, dataloader_trn, args, VIT_name, datapath)



