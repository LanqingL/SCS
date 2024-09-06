import os
import math
import time
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import h5py
import torch.nn as nn
from typing import Tuple
from datetime import datetime
from torch import Tensor as T
from .ICL import models_seggpt
from .ICL.seggpt_engine import inference_image
from .cluster_based_policy_feature import cluster_to_offline_nearest
from .UTILS import calculate_metric, get_query_support, prepare_model, round_image, WHITE, BLACK, Logger, to_cuda


def policy_gradient_train(policy_model, datasets, args, VIT_name, datapath):

    if args.resume:
        projector_checkpoint = torch.load(args.ckpt_projector_path)
        predictor_checkpoint = torch.load(args.ckpt_predictor_path)
        policy_model.projector.load_state_dict(projector_checkpoint)
        policy_model.predictor.load_state_dict(predictor_checkpoint)

    device = torch.device(args.device)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)  # backbone + agent
    model = prepare_model(models_seggpt, args.engine_ckpt_path, args.model, args.seg_type).to(device)  # generalist segmentation model

    logger = Logger(os.path.join(args.output_path, 'log.txt'))

    reward_history = []
    loss_history = []

    total_reward_history = []  # epoch based
    total_loss_history = []  # epoch based

    STOP_FLAG = False

    temp_min = 0.5
    ANNEAL_RATE = 0.00003

    for epoch in range(args.epochs):
        logger.write(f"Epoch: {epoch}/{args.epochs}")

        total_train_reward = 0
        total_train_loss = 0
        total_iter = math.ceil(len(datasets.batch_sampler.sampler.data_source.img_metadata) / args.batch_size)
        temp = args.temp
        for idx, batch in enumerate(datasets):
            start_time = time.time()

            batch = to_cuda(batch)
            query_image, query_name, class_sample = batch['query_img'], batch['query_name'], batch['class_id']
            all_support_names = batch['support_names']

            # len(input_list) x hidden_size
            embedding_query = policy_model.model.forward_features(query_image.to(device))
            embedding_query = policy_model.model.forward_head(embedding_query, pre_logits=True)
            embedding_query = (embedding_query + policy_model.projector(embedding_query)).unsqueeze(1)

            all_support_path = []
            for support_id in range(len(all_support_names[0])):
                all_support_path.append(list(list(zip(*all_support_names))[support_id]))
            scores = []
            embedding_names = []
            all_support_features = []
            for id_batch in range(len(query_image)):
                can_features = []
                can_images_path = []
                is_find_id = -1
                with h5py.File(VIT_name, 'r') as f:
                    class_sample_id = class_sample.cpu().numpy()[id_batch]
                    class_sample_dataset = f[str(class_sample_id)]
                    choose_support_name = class_sample_dataset['cluster_result']['all_image_name'][:]
                    cluster_id = class_sample_dataset['cluster_result']['cluster_id'][:]
                    cluster_centers = class_sample_dataset['cluster_result']['cluster_centers'][:]

                    num_id = 0  ####
                    show_support_name = []

                    query_select_name = os.path.basename(query_name[id_batch])
                    for chose_id in choose_support_name:  # if query in set, mark it
                        chose_id = chose_id.decode('utf-8')
                        if chose_id == query_select_name:
                            is_find_id = num_id

                        image_name = chose_id
                        sample_dataset = class_sample_dataset[image_name]

                        if sample_dataset.__contains__(str(0)):
                            now_chose_id = show_support_name.count(chose_id)
                            can_features.append(sample_dataset[str(now_chose_id)]['features'][:])
                            can_images_path.append(sample_dataset[str(now_chose_id)]['path'][:])
                        else:
                            can_features.append(sample_dataset['features'][:])
                            can_images_path.append(sample_dataset['path'][:])
                        num_id = num_id + 1


                if is_find_id == -1:  # if not query in set, add it at the end
                    add_query_feature = policy_model.model.forward_features(query_image[id_batch].unsqueeze(0).to(device))
                    add_query_feature = policy_model.model.forward_head(add_query_feature, pre_logits=True)  # len(input_list) x hidden_size

                    add_query_feature = add_query_feature.cpu().detach().numpy()
                    can_features.append(add_query_feature)
                    is_find_id = len(can_features) - 1

                can_features = np.array(can_features)
                if len(can_features.shape) > 2:
                    can_features = can_features.squeeze(1)
                sample_ids = cluster_to_offline_nearest(can_features, cluster_id,
                                                        cluster_centers, device, is_find_id, args)




                support_path_id = [can_images_path[sample_id][0].decode('utf-8') for sample_id in sample_ids]
                support_sample_features = [can_features[sample_id] for sample_id in sample_ids]
                support_sample_features = torch.from_numpy(np.stack(support_sample_features, axis=0)).to(device)

                support_sample_features = support_sample_features + policy_model.projector(support_sample_features)
                support_sample_features = policy_model.predictor(support_sample_features)

                all_support_features.append(support_sample_features)
                embedding_names.append(support_path_id)

                query_norm = nn.functional.normalize(embedding_query[id_batch], dim=1)
                support_norm = nn.functional.normalize(support_sample_features, dim=1)
                score = torch.mm(query_norm,
                                 support_norm.permute(1, 0)) / args.Tem  # [1, len(cand_examples)]

                scores.append(score)


            output_scores = []
            for id_batch in range(len(query_image)):
                output_score = torch.nn.functional.gumbel_softmax(scores[id_batch])
                output_scores.append(output_score)

            reward, loss = get_batch_reward_loss(output_scores, query_name,
                                                 embedding_names, class_sample, model, device, args)


            end_time = time.time()
            batch_time = end_time - start_time
            logger.write(f"{datetime.now()} Epoch: {epoch}/{args.epochs}, Batch: {idx}/{total_iter},"
                         f" reward: {reward:.3f}, loss: {loss:.3f}, time_cost:{batch_time:.3f}")


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * idx), temp_min)

            # for each iteration/batch
            total_train_reward += reward
            total_train_loss += loss.item()

            reward_history.append(reward)
            loss_history.append(loss.item())

            if np.isnan(loss.item()):
                STOP_FLAG = True
                break

        # for each epoch
        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)

        best_reward_epoch = total_reward_history.index(best_reward)
        best_loss_epoch = total_loss_history.index(best_loss)

        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(f"### Total reward: {total_train_reward}, " + f"Total loss: {round(total_train_loss,5)}, " +
                     f"Best reward: {best_reward} at epoch {best_reward_epoch}, " +
                     f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n")

        # save every epoch
        ckpt_projector_file = os.path.join(args.output_path, f"{args.BENCHMARK}_projector_{epoch}.pt")
        torch.save(policy_model.projector.state_dict(), ckpt_projector_file)
        ckpt_predictor_file = os.path.join(args.output_path, f"{args.BENCHMARK}_predictor_{epoch}.pt")
        torch.save(policy_model.predictor.state_dict(), ckpt_predictor_file)
        logger.write(f"saved the {epoch} ckpt to {ckpt_projector_file}")
        logger.write(f"saved the {epoch} ckpt to {ckpt_predictor_file}")

        # save best epoch
        if epoch == best_reward_epoch:
            ckpt_projector_file = os.path.join(args.output_path, f"{args.BENCHMARK}_best_projector.pt")
            torch.save(policy_model.projector.state_dict(), ckpt_projector_file)
            ckpt_predictor_file = os.path.join(args.output_path, f"{args.BENCHMARK}_best_predictor.pt")
            torch.save(policy_model.predictor.state_dict(), ckpt_predictor_file)
            logger.write(f"saved the best reward ckpt to {ckpt_projector_file}")
            logger.write(f"saved the best reward ckpt to {ckpt_predictor_file}")

        if STOP_FLAG:
            break

    # save in the end
    ckpt_projector_file = os.path.join(args.output_path, f"{args.BENCHMARK}_final_projector.pt")
    torch.save(policy_model.projector.state_dict(), ckpt_projector_file)
    ckpt_predictor_file = os.path.join(args.output_path, f"{args.BENCHMARK}_final_predictor.pt")
    torch.save(policy_model.predictor.state_dict(), ckpt_predictor_file)

def get_batch_reward_loss(scores, query_name, support_names, class_sample, model, device, args):

    total_batch_num = 0
    batch_reward = 0
    all_scores = []
    all_ious = []

    ## loop over the training examples
    mean_ious = []
    for i in range(len(scores)):
        # interact with the environment to get rewards, which in our case is to feed the prompt into GPT-3 and evaluate the prediction
        cand_prob = scores[i].squeeze(0).clone().detach()
        cand_prob = cand_prob.cpu().numpy()
        cand_prob = np.nan_to_num(cand_prob, nan=0.0001)  # replace np.nan with 0
        cand_prob /= cand_prob.sum()  # make probabilities sum to 1

        query_image, query_mask, support_image, support_masks, cids = get_query_support(query_name[i], class_sample[i],
                                                                                        cand_prob, support_names[i], args)

        current_metrics = []
        ALL_IOU = []

        mean_iou = 0
        for cid in cids:
            sin_support_image, sin_support_masks = support_image[cid], support_masks[cid]
            query_image_name, sin_support_image = query_image, sin_support_image
            output_image_mask, output, _, _ = inference_image(model, device, query_image_name,
                                                              [sin_support_image], [sin_support_masks])
            if isinstance(query_mask, Image.Image):
                image_mask = query_mask.convert("RGB")
            else:
                image_mask = Image.open(query_mask).convert("RGB")
            # input_image = np.array(image_mask)
            image_mask = np.uint8(np.array(image_mask))

            image_mask = round_image(image_mask, [WHITE, BLACK])
            output_image_mask = round_image(output_image_mask, [WHITE, BLACK], t=args.t)
            current_metric = calculate_metric(image_mask, output_image_mask)

            current_metrics.append(current_metric)
            ALL_IOU.append(current_metric["iou"])
            mean_iou = mean_iou + current_metric["iou"]

        mean_ious.append(mean_iou/len(cids))
        all_scores.append(scores[i][0][cids].unsqueeze(0))
        all_ious.append(ALL_IOU)


        _reward = torch.tanh(torch.from_numpy(np.array((1 - current_metrics[0]['iou']))))
        batch_reward = batch_reward + torch.tanh(torch.from_numpy(np.array((current_metrics[0]['iou']))))
        total_batch_num = total_batch_num + 1


    batch_iou_loss = 0
    batch_iou = 0
    for idx in range(len(all_ious)):
        for selective_idx in range(args.sample_num):
            batch_iou += all_ious[idx][selective_idx] / args.sample_num

            batch_iou_loss -= ((all_ious[idx][selective_idx] - mean_ious[idx]) * torch.log(
                all_scores[idx][0][selective_idx]))

    batch_iou_loss = batch_iou_loss / total_batch_num
    batch_iou = batch_iou / total_batch_num

    return batch_iou, batch_iou_loss


class BiEncoderNllLoss(object):
    def calc(
        self,
        SCORES: T,
        DOWNSTREAM_IOUS: list,
    ) -> Tuple[T, T]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        positive_idx = []
        for DOWNSTREAM_IOU in DOWNSTREAM_IOUS:
            DOWNSTREAM_IDX = sorted(range(len(DOWNSTREAM_IOU)), key=lambda i: DOWNSTREAM_IOU[i], reverse=True)

            positive_idx.append(torch.tensor([DOWNSTREAM_IDX[0]]).to(SCORES.device))

        positive_idx = torch.cat(tuple(positive_idx), dim=0)
        softmax_scores = F.log_softmax(SCORES, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            positive_idx,
            reduction="mean",
        )

        return loss, positive_idx
