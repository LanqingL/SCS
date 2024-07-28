import json
import argparse
import random
import time
import copy
import torch
import h5py
import torchvision
import numpy as np
import torch.nn as nn
import albumentations as A
from datetime import datetime
import torch.nn.functional as F

from PIL import Image
from data.dataset import FSSDataset
from data.dataset import Compose
from RL_base.model import policy_network
from RL_base.UTILS import WHITE, round_image, BLACK, Logger
# import RL_base.UTILS as utils
from RL_base.UTILS import prepare_save_vit_h5py, prepare_cluster_h5py, prepare_cand_h5py
from RL_base.UTILS import calculate_metric, prepare_model, add_image_path
from RL_base.ICL.seggpt_engine import inference_image
from RL_base.ICL import models_seggpt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='Prompt num')
    parser.add_argument('--model_name',
                        type=str,
                        default='vit_large_patch14_clip_224.laion2b_ft_in12k_in1k',
                        help='Policy network Pre-training model.')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default='./pretrained_model/pytorch_model.bin',
                        help='Policy network Pre-training model checkpoint path.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--base_dir', default='./datasets',
                        help='pascal base dir')
    parser.add_argument('--fold', default=1, type=int)
    # parser.add_argument('--split', default='val', type=str)
    parser.add_argument('--flip', default=0, type=int)
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
    parser.add_argument('--BENCHMARK', default='pascal', type=str, help='dataset type:"pascal, fss, coco, isald"')
    parser.add_argument('--cats_augmentation', default=True, type=bool)
    parser.add_argument('--pfenet_augmentation', default=False, type=bool)
    parser.add_argument('--candidate_num', default=50, type=int, help='candidate examples num')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--output_root', type=str, default='./log/inference')
    parser.add_argument('--engine_ckpt_path', type=str, help='path to ckpt',
                        default='./pretrained_model/seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types',
                        choices=['instance', 'semantic'], default='semantic')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--cluster_num', type=int, default=10,
                        help='number of cluster')
    parser.add_argument('--Tem', default=1, type=int, help='Temperature')
    parser.add_argument('--backbone', default="VIT", type=str, help='backbone of retrival feature')
    parser.add_argument('--ckpt_projector_path',
                        type=str,
                        default="./log/pascal/pascal_best_projector.pt",
                        help='project ckpt relative path')
    parser.add_argument('--ckpt_predictor_path',
                        type=str,
                        default="./log/pascal/pascal_best_predictor.pt",
                        help='predictor ckpt relative path')

    parser.add_argument('--mode', default='val', type=str, help='train, val')

    args = parser.parse_args()
    return args

def retrieve_from_candiate_pool(VIT_name, class_sample, query_name, args):
    with h5py.File(VIT_name, 'r') as f:
        class_sample_id = class_sample.cpu().numpy()

        class_sample_dataset = f[str(class_sample_id)]
        choose_support_name = copy.deepcopy(
            class_sample_dataset['selected_result']['selected_image_name'][:])
        chose_support_features = copy.deepcopy(
            class_sample_dataset['selected_result']['selected_features'][:])

        query_select_name = os.path.basename(query_name)

        end_support_name = []
        end_support_feature = []
        for chose_id in range(len(choose_support_name)):  # if query in set, mark it
            chose_name = choose_support_name[chose_id].decode('utf-8')
            if chose_name == query_select_name:
                continue
            end_support_name.append(choose_support_name[chose_id])
            end_support_feature.append(chose_support_features[chose_id])

        return end_support_feature, end_support_name


if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    # Dataset initialization
    datapath = os.path.join(args.base_dir, 'pascal-5i')

    # policy network
    policy_model = policy_network(model_name=args.model_name,
                                  add_linear=True,
                                  embedding_size=args.embedding_size,
                                  checkpoint_path=args.ckpt_path,
                                  freeze_encoder=True)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    policy_model = policy_model.to(device)

    model = policy_model.model

    seg_model = prepare_model(models_seggpt, args.engine_ckpt_path, args.model, args.seg_type).to(device)

    train_Img_size = model.pretrained_cfg['input_size'][1]
    FSSDataset.initialize(benchmark=args.BENCHMARK, img_size=train_Img_size, datapath=datapath, use_original_imgsize=False,
        apply_cats_augmentation=args.cats_augmentation, apply_pfenet_augmentation=args.pfenet_augmentation)
    dataloader_val = FSSDataset.build_dataloader(args.BENCHMARK, args.batch_size, args.num_workers, args.fold, 'val', shot=args.candidate_num)
    # ======================================================= INFERENCE ===============================================
    if os.path.exists(args.ckpt_projector_path) and os.path.exists(args.ckpt_predictor_path):
        policy_model.projector.load_state_dict(torch.load(args.ckpt_projector_path))
        policy_model.predictor.load_state_dict(torch.load(args.ckpt_predictor_path))
    else:
        print(f"The ckpt path for [{args.ckpt_projector_path} or {args.ckpt_predictor_path}] does not exist!")  # CHECK
        exit()

    policy_model.eval()


    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    img_metadata_classwise = dataloader_val.sampler.data_source.img_metadata_classwise
    can_supports = {}
    for class_sample in range(len(img_metadata_classwise)):
        class_sample_length = len(img_metadata_classwise[class_sample])
        if class_sample_length == 0:
            continue
        can_supports[class_sample] = img_metadata_classwise[class_sample]


    transform = Compose([
        A.Resize(train_Img_size, train_Img_size),
        A.Normalize(img_mean, img_std),
        A.pytorch.transforms.ToTensorV2(),
    ])

    VIT_root = os.path.join(args.output_root, args.BENCHMARK)
    os.makedirs(VIT_root, exist_ok=True)
    VIT_name = os.path.join(VIT_root, args.BENCHMARK + str(args.fold) + '-test.h5')

    if os.path.exists(VIT_name):
        pass
    else:
        prepare_save_vit_h5py(VIT_name, can_supports, datapath, policy_model, transform, device, args)

        if args.offline_cluster:
            prepare_cluster_h5py(VIT_name, can_supports.keys(), can_supports, device, args)
    prepare_cand_h5py(VIT_name, can_supports.keys(), can_supports, args)

    with torch.no_grad():
        padding = 1
        image_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
             torchvision.transforms.ToTensor()])
        mask_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
             torchvision.transforms.ToTensor()])

        img_path = os.path.join(args.base_dir, 'pascal-5i/VOC2012/JPEGImages/')
        ann_path = os.path.join(args.base_dir, 'pascal-5i/VOC2012/SegmentationClassAug/')
        total_sample_num = 1000
        cand_embeddings = {}

        logger = Logger(os.path.join(args.output_root, str(args.BENCHMARK) + '-' + str(args.cluster_num)
                                     + '-' + str(args.shot) + 'log.txt'))
        miou, m_color_blind_iou, m_accuracy = 0, 0, 0
        mbatch_time = 0
        for idx, batch in enumerate(dataloader_val):
            # 1. Preprocess
            start_time = time.time()
            now_element = batch
            query_image, query_name, class_sample = batch['query_img'], batch['query_name'], batch['class_id']

            embedding_query = policy_model.model.forward_features(query_image.to(device))
            embedding_query = policy_model.model.forward_head(embedding_query, pre_logits=True)

            assert len(query_image) == 1

            for id_batch in range(len(query_image)):
                end_support_feature, end_support_name = retrieve_from_candiate_pool(VIT_name,
                                                                                    class_sample[id_batch],
                                                                                    query_name[id_batch], args)

            support_path_id = [end_support_name[sample_id].decode('utf-8')
                               for sample_id in range(len(end_support_name))]
            support_sample_features = [end_support_feature[sample_id]
                                       for sample_id in range(len(end_support_feature))]

            assert len(support_sample_features) == len(support_path_id)

            # 2. search visual prompts
            embedding_query = (policy_model.projector(embedding_query))
            support_sample_features = torch.from_numpy(np.stack(support_sample_features, axis=0)).to(device)
            support_sample_features = policy_model.predictor(policy_model.projector(support_sample_features))

            query_norm = nn.functional.normalize(embedding_query, dim=1)
            support_norm = nn.functional.normalize(support_sample_features, dim=1)
            scores = torch.mm(query_norm,
                              support_norm.permute(1, 0)) / args.Tem  # [1, len(cand_examples)]

            scores = F.softmax(scores, dim=1)[0]  # [cand_num]

            scores = scores.cpu().detach().numpy().tolist()
            cand_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.shot]

            chose_support_name = [support_path_id[chose_id] for chose_id in cand_ids]


            # 3. Inference
            query_image_name, support_image, query_mask, support_masks = add_image_path(datapath, img_path,
                                                                                        ann_path,
                                                                                        query_name,
                                                                                        chose_support_name,
                                                                                        class_sample, args)

            output_image_mask, output, _, _ = inference_image(seg_model, device,
                                                              query_image_name, support_image, support_masks)
            if isinstance(query_mask, Image.Image):
                image_mask = query_mask.convert("RGB")
            else:
                image_mask = Image.open(query_mask).convert("RGB")

            # 4. Evaluate prediction
            image_mask = np.uint8(np.array(image_mask))
            image_mask = round_image(image_mask, [WHITE, BLACK])
            output_image_mask = round_image(output_image_mask, [WHITE, BLACK], t=args.t)

            current_metric = calculate_metric(image_mask, output_image_mask)

            end_time = time.time()
            batch_time = end_time - start_time
            logger.write(f"{datetime.now()} Batch: {idx}/{total_sample_num}, iou: {current_metric['iou']:.3f},"
                         f" 'accuracy':{current_metric['accuracy']:.3f}, time_cost:{batch_time:.3f}")
            logger.write(f"Query image name: {query_name[0]}, class_id:{class_sample.cpu().numpy()[0]}")
            logger.write(f"Support image name:{chose_support_name}, Support_class_id:{class_sample.cpu().numpy()[0]}")

            miou += current_metric['iou']
            m_accuracy += current_metric['accuracy']
            mbatch_time += batch_time

        logger.write(f"mIOU:{(miou / total_sample_num)},"
                     f" m_accuracy:{(m_accuracy / total_sample_num)},"
                     f" mour_time:{mbatch_time / total_sample_num}")




