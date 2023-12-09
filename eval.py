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
from model.fashionnet import get_net

from reproducible_code.tools.io import save_txt, load_txt
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

    LOGGER.info(
        "Average accuracy: %.3f, Binary Accuracy: %.3f", accuracy, binary
    )

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
        scores = [[[] for _ in range(num_users)] for _ in range(4)]
        u = 0  # 1 user
        for inputs in tqdm(loader, desc="Computing scores"):
            inputs = utils.to_device(inputs, device)
            with torch.no_grad():
                outputs, _, _ = net.visual_output(**inputs)
                outputs = [s.tolist() for s in outputs]

            for u in range(num_users):
                for n, score in enumerate(outputs):
                    for s in score:
                        scores[n][u].append(s)  # [N, U, S, 1]
        return scores

    parallel, device = utils.get_device(config.gpus)
    LOGGER.info("Testing for NDCG and AUC.")

    net = get_net(config, logger)
    net.eval()

    data_param = config.data_param
    data_param.shuffle = False
    LOGGER.info("Dataset for positive tuples: %s", data_param)
    loader = get_dataloader(data_param, logger)
    loader.make_nega()

    scores = outfit_scores()

    # compute ndcg
    mean_ndcg, avg_ndcg = utils.metrics.NDCG(scores[0], scores[2])
    mean_ndcg_binary, avg_ndcg_binary = utils.metrics.NDCG(
        scores[1], scores[3]
    )
    aucs, mean_auc = utils.metrics.ROC(scores[0], scores[2])
    aucs_binary, mean_auc_binary = utils.metrics.ROC(scores[1], scores[3])
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


# TODO: Check fitb
def fitb(config, logger):
    parallel, device = utils.get_device(config.gpus)
    param = config.fitb_data_param
    LOGGER.info("Get data for FITB questions: %s", param)
    loader = get_dataloader(param, loggsr)
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
                scores, _ = net.visual_output(**inputs)
        scores = scores[0][1]
        # the first item is the groud-truth item
        if torch.argmax(scores).item() == 0:
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
