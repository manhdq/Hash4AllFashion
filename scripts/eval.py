#!/usr/bin/env python
"""Script for training, evaluate and retrieval."""
import argparse
import logging
import os
import os.path as osp
import pickle
import shutil
import textwrap

import numpy as np
import torch
from tqdm import tqdm
import yaml
from torch.nn.parallel import data_parallel

from dataset import get_dataloader

import utils
from utils import param
from utils.logger import get_logger
from model import get_net

from icecream import ic


def update_npz(fn, results):
    os.makedirs(osp.dirname(fn), exist_ok=True)    
    if fn is None:
        return
    if osp.exists(fn):
        pre_results = dict(np.load(fn, allow_pickle=True))
        pre_results.update(results)
        results = pre_results
    np.savez(fn, **results)


def evalute_accuracy(config, logger):
    """Evaluate fashion net for accuracy."""
    # make data loader
    parallel, device = utils.get_device(config.gpus)
    param = config.data_param
    loader = get_dataloader(param, logger)
    pbar = tqdm(loader)
    net = get_net(config, logger)
    net.eval()

    ic(loader.num_sample)

    # set data mode to pair for testing pair-wise accuracy
    LOGGER.info("Testing for accuracy")

    num_sample = loader.num_sample
    batchs = accuracy = binary = 0.0

    for idx, inputs in enumerate(pbar):
        # compute output and loss
        batch_size = inputs["outf_s"].shape[0]
        ic(batch_size)
        inputs = utils.to_device(inputs, device)
        with torch.no_grad():
            if parallel:
                output = data_parallel(net, inputs, config.gpus)
            else:
                output = net(**inputs)
        _, batch_results = net.gather(output)
        batch_accuracy = batch_results["accuracy"]
        batch_binary = batch_results["binary_accuracy"]
        
        # LOGGER.info(
        #     "Batch [%d]/[%d] Accuracy %.3f Accuracy (Binary Codes) %.3f",
        #     idx,
        #     loader.num_batch,
        #     batch_accuracy,
        #     batch_binary,
        # )

        accuracy += batch_accuracy * batch_size
        binary += batch_binary * batch_size
        batchs += batch_size
        
    accuracy /= num_sample
    binary /= num_sample
    assert batchs == num_sample, ic(batchs)

    LOGGER.info("Average accuracy: %.3f, Binary Accuracy: %.3f", accuracy, binary)

    # save results
    if net.param.zero_iterm:
        results = dict(uaccuracy=accuracy, ubinary=binary)
    elif net.param.zero_uterm:
        results = dict(iaccuracy=accuracy, ibinary=binary)
    else:
        results = dict(accuracy=accuracy, binary=binary)
    update_npz(config.result_file, results)


def evalute_rank(config, logger):
    """Evaluate fashion net for NDCG an AUC."""

    def outfit_scores():
        """Compute rank scores for data set."""
        num_users = net.param.num_users
        scores = [[] for u in range(num_users)]
        binary = [[] for u in range(num_users)]
        for inputs in tqdm(loader, desc="Computing scores"):
            inputs = utils.to_device(inputs, device)
            with torch.no_grad():
                if parallel:
                    output = data_parallel(net, inputs, config.gpus)
                else:
                    scores, _, _ = net.visual_output(**inputs)
            for n, s in enumerate(scores):
                scores[u].append(output[0][n].item())
                binary[u].append(output[1][n].item())
        return scores, binary

    parallel, device = utils.get_device(config.gpus)
    LOGGER.info("Testing for NDCG and AUC.")

    net = get_net(config, logger)
    net.eval()

    data_param = config.data_param
    data_param.shuffle = False
    LOGGER.info("Dataset for positive tuples: %s", data_param)
    loader = get_dataloader(data_param, logger)
    loader.make_nega()
    loader.set_data_mode("PosiOnly")
    posi_score, posi_binary = outfit_scores()
    LOGGER.info("Compute scores for positive outfits, done!")
    loader.set_data_mode("NegaOnly")
    nega_score, nega_binary = outfit_scores()
    LOGGER.info("Compute scores for negative outfits, done!")

    # compute ndcg
    mean_ndcg, avg_ndcg = utils.metrics.NDCG(posi_score, nega_score)
    mean_ndcg_binary, avg_ndcg_binary = utils.metrics.NDCG(posi_binary, nega_binary)
    aucs, mean_auc = utils.metrics.ROC(posi_score, nega_score)
    aucs_binary, mean_auc_binary = utils.metrics.ROC(posi_binary, nega_binary)
    LOGGER.info(
        "Metric:\n"
        "- average ndcg:%.4f\n"
        "- average ndcg(binary):%.4f\n"
        "- mean auc:%.4f\n"
        "- mean auc(binary):%.4f",
        mean_ndcg.mean(),
        mean_ndcg_binary.mean(),
        mean_auc,
        mean_auc_binary,
    )

    # save results
    results = dict(
        posi_score_binary=posi_binary,
        posi_score=posi_score,
        nega_score_binary=nega_binary,
        nega_score=nega_score,
        mean_ndcg=mean_ndcg,
        avg_ndcg=avg_ndcg,
        mean_ndcg_binary=mean_ndcg_binary,
        avg_ndcg_binary=avg_ndcg_binary,
        aucs=aucs,
        mean_auc=mean_auc,
        aucs_binary=aucs_binary,
        mean_auc_binary=mean_auc_binary,
    )
    update_npz(config.result_file, results)

    # saved ranked outfits
    result_dir = config.result_dir
    if config.result_dir is None:
        return
    assert not data_param.variable_length
    labels = [
        np.array([1] * len(pos) + [0] * len(neg))
        for pos, neg in zip(posi_score, nega_score)
    ]
    outfits = loader.dataset.get_outfits_list()
    sorting = [
        np.argsort(-1.0 * np.array(pos + neg))
        for pos, neg in zip(posi_binary, nega_binary)
    ]
    utils.check.check_dirs(result_dir, action="mkdir")
    ndcg_fn = os.path.join(result_dir, "ndcg.txt")
    label_folder = os.path.join(result_dir, "label")
    outfit_folder = os.path.join(result_dir, "outfit")
    utils.check.check_dirs([label_folder, outfit_folder], action="mkdir")
    np.savetxt(ndcg_fn, mean_ndcg_binary)
    for uid, ranked_idx in tqdm.tqdm(enumerate(sorting), desc="Computing outfits"):
        # u is the user id, rank is the sorting for outfits
        folder = os.path.join(outfit_folder, "user-%03d" % uid)
        utils.check.check_dirs(folder, action="mkdir")
        label_file = os.path.join(label_folder, "user-%03d.txt" % uid)
        # save the rank list for current user
        np.savetxt(label_file, labels[uid][ranked_idx], fmt="%d")
        # rank the outfit according to rank scores
        for n, idx in enumerate(ranked_idx):
            # tpl is the n-th ranked outfit
            tpl = outfits[uid][idx]
            y = labels[uid][idx]
            image_folder = os.path.join(folder, "top-%03d-%d" % (n, y))
            utils.check.check_dirs(image_folder, action="mkdir")
            for cate, item_id in enumerate(tpl):
                src = loader.dataset.get_image_path(cate, item_id)
                dst = os.path.join(image_folder, "%02d.jpg" % cate)
                shutil.copy2(src, dst)
    LOGGER.info("All outfits are save in %s", config.result_dir)


# TODO: Check fitb
def fitb(config, logger):
    parallel, device = utils.get_device(config.gpus)
    data_param = param.fitb_data_param
    LOGGER.info("Get data for FITB questions: %s", data_param)
    loader = get_dataloader(data_param)
    pbar = tqdm.tqdm(loader, desc="Computing scores")
    net = get_net(config, logger)
    net.eval()
    correct = 0
    cnt = 0

    for inputs in pbar:
        inputs = utils.to_device(inputs, device)
        with torch.no_grad():
            # TODO: Check parallel
            if parallel:
                _, score_b = data_parallel(net, inputs, config.gpus)
            else:
                _, score_b = net(**inputs)
        # the first item is the groud-truth item
        if torch.argmax(score_b).item() == 0:
            correct += 1
        cnt += 1
        pbar.set_description("Accuracy: {:.3f}".format(correct / cnt))
    fitb_acc = correct / cnt
    LOGGER.info("FITB Accuracy %.4f", fitb_acc)
    results = dict(fitb_acc=fitb_acc)
    update_npz(config.result_file, results)



ACTION_FUNS = {
    "fitb": fitb,
    "evaluate-accuracy": evalute_accuracy,
    "evaluate-rank": evalute_rank,
}

LOGGER = logging.getLogger("fashion32")
LOGGER.setLevel(logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Fashion Hash Net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Fashion Hash Net Evaluation Script
            --------------------------------
            Actions:
                1. train: train fashion net.
                2. evaluate: evaluate NDCG and accuracy.
                3. retrieval: retrieval for items.
                """
        ),
    )
    actions = ACTION_FUNS.keys()
    parser.add_argument("action", help="|".join(sorted(actions)))
    parser.add_argument("--cfg", help="configuration file.")
    parser.add_argument(
        "--env",
        default="local",
        choices=["local", "colab"],
        help="Using for logging option. Using logger if local, using normal print otherwise.",
    )
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        kwargs = yaml.load(f, Loader=yaml.FullLoader)
    config = param.FashionParam(**kwargs)
    config.add_timestamp()

    logger = get_logger(args.env, config)
    logger.info(f"Fashion param : {config}")    

    if args.action in actions:
        ACTION_FUNS[args.action](config, logger)
        exit(0)
    else:
        LOGGER.info("Action %s is not in %s", args.action, "|".join(actions))
        exit(1)
