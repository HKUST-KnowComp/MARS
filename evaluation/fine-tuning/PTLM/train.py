import argparse
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from dataloader import EventDiscriminationDataset
from evaluate import evaluate
from model import MetaphysicalEventDiscriminator

sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # model related
    group_model = parser.add_argument_group("model configs")
    group_model.add_argument("--ptlm", default='microsoft/deberta-v3-large', type=str,
                             required=False, help="choose from huggingface PTLM")
    group_model.add_argument("--pretrain_model_from_path", required=False, default="",
                             help="pretrain this model from a checkpoint")
    group_model.add_argument("--pretrain_tokenizer_from_path", required=False, default="",
                             help="pretrain tokenizer from a checkpoint")

    # training-related args
    group_trainer = parser.add_argument_group("training configs")

    group_trainer.add_argument("--device", default='cuda', type=str, required=False,
                               help="device")
    group_trainer.add_argument("--gpu", default=1, type=int, required=False,
                               help="gpu number")
    group_trainer.add_argument("--optimizer", default='ADAMW', type=str, required=False,
                               help="optimizer")
    group_trainer.add_argument("--lr", default=3e-6, type=float, required=False,
                               help="learning rate")
    group_trainer.add_argument("--batch_size", default=64, type=int, required=False,
                               help="batch size")
    group_trainer.add_argument("--test_batch_size", default=32, type=int, required=False,
                               help="test batch size")
    group_trainer.add_argument("--epochs", default=5, type=int, required=False,
                               help="number of epochs")
    group_trainer.add_argument("--max_length", default=100, type=int, required=False,
                               help="max_seq_length of h+r+t")
    group_trainer.add_argument("--eval_every", default=25, type=int, required=False,
                               help="eval on test set every x steps.")

    # IO-related
    group_data = parser.add_argument_group("IO related configs")
    group_data.add_argument("--output_dir", default="results-CANDLE", type=str, required=False,
                            help="where to output.")
    group_data.add_argument("--experiment_name", default='no_boundary', type=str, required=False,
                            help="A special name that will be prepended to the dir name of the output.")

    group_data.add_argument("--seed", default=829, type=int, required=False, help="random seed")

    args = parser.parse_args()

    return args


def main():
    # get all arguments
    args = parse_args()

    # Check device status
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA available:', torch.cuda.is_available())
    print(torch.cuda.get_device_name())
    print('Device number:', torch.cuda.device_count())
    print(torch.cuda.get_device_properties(device))
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.cuda.set_device(args.gpu)

    experiment_name = args.experiment_name

    save_dir = os.path.join(args.output_dir, "_".join([os.path.basename(args.ptlm),
                                                       f"bs{args.batch_size}", f"lr{args.lr}",
                                                       f"evalstep{args.eval_every}"]) + '_' + experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("metaphysical_transition_discrimination-{}".format(args.ptlm))
    handler = logging.FileHandler(os.path.join(save_dir, f"log_seed_{args.seed}.txt"))
    logger.addHandler(handler)

    # set random seeds
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # load model
    model = MetaphysicalEventDiscriminator(args.ptlm).to(args.device)

    if args.pretrain_tokenizer_from_path:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_tokenizer_from_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.ptlm)

    if 'gpt2' in args.ptlm:
        model = GPT2LMHeadModel.from_pretrained(args.ptlm).to(args.device)

        tokenizer = GPT2Tokenizer.from_pretrained(args.ptlm)

    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]'})
    tokenizer.add_tokens(['<c>', '</c>'])
    if 'gpt2' in args.ptlm:
        model.resize_token_embeddings(len(tokenizer))
    else:
        model.model.resize_token_embeddings(len(tokenizer))

    if args.pretrain_model_from_path:
        ckpt = torch.load(args.pretrain_model_from_path)
        model.load_state_dict(ckpt)

    # load metaphysical_event_data
    if args.experiment_name == 'fixed_boundary':
        train_dataset = pd.read_csv('../../data_with_equal_boundary/train.csv', index_col=None)
        dev_dataset = pd.read_csv('../../data_with_equal_boundary/dev.csv', index_col=None)
        test_dataset = pd.read_csv('../../data_with_equal_boundary/test.csv', index_col=None)
    else:
        train_dataset = pd.read_csv('../../data/train.csv', index_col=None)
        dev_dataset = pd.read_csv('../../data/dev.csv', index_col=None)
        test_dataset = pd.read_csv('../../data/test.csv', index_col=None)

    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 5
    }

    val_params = {
        'batch_size': args.test_batch_size,
        'shuffle': False,
        'num_workers': 5
    }

    training_set = EventDiscriminationDataset(dataframe=train_dataset, tokenizer=tokenizer, max_length=args.max_length,
                                              model=args.ptlm, split='trn')
    validation_set = EventDiscriminationDataset(dataframe=dev_dataset, tokenizer=tokenizer,
                                                max_length=args.max_length, model=args.ptlm, split='dev')
    testing_set = EventDiscriminationDataset(dataframe=test_dataset, tokenizer=tokenizer, max_length=args.max_length,
                                             model=args.ptlm, split='tst')

    training_loader = DataLoader(training_set, **train_params, drop_last=True)
    dev_dataloader = DataLoader(validation_set, **val_params, drop_last=False)
    tst_dataloader = DataLoader(testing_set, **val_params, drop_last=False)

    # model training
    if args.optimizer == "ADAM":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    elif args.optimizer == 'ADAMW':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()
    best_val_auc, best_val_acc, best_val_macroF1, best_tst_auc, best_tst_acc, best_tst_macroF1 = 0, 0, 0, 0, 0, 0

    model.train()
    progress_bar = tqdm(range(len(training_loader) * args.epochs), desc="Training")

    iteration = 0
    for e in range(args.epochs):

        for iteration, data in enumerate(training_loader, iteration + 1):
            # the iteration starts from 1.

            y = data['label'].to(args.device, dtype=torch.long)

            ids = data['ids'].to(args.device, dtype=torch.long)
            mask = data['mask'].to(args.device, dtype=torch.long)
            tokens = {"input_ids": ids, "attention_mask": mask}

            if 'gpt2' in args.ptlm:
                outputs = model(input_ids=ids, attention_mask=mask, labels=ids)
                loss = outputs[0]
            else:
                logits = model(tokens)
                loss = criterion(logits, y)
            # print(f"Epoch {e} Step {iteration} Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)

            if args.eval_every > 0 and iteration % args.eval_every == 0:
                model.eval()

                eval_auc, eval_macro_f1, eval_acc, _ = evaluate(tokenizer, model, args.device,
                                                                dev_dataloader, 0.5, model_type=args.ptlm)
                assert _ == len(dev_dataset)
                print('Current dev eval: AUC ', eval_auc, 'Best AUC: ', best_val_auc)
                print('Current dev eval: ACC ', eval_acc, 'Best ACC: ', best_val_acc)
                print('Current dev eval: Macro F1 ', eval_macro_f1, 'Best Tuned_ACC: ', best_val_macroF1)
                updated = []
                if eval_auc > best_val_auc:
                    updated.append("AUC")
                    best_val_auc = eval_auc
                    # torch.save(model.state_dict(), save_dir + f"/best_val_auc_model_seed_{args.seed}.pth")
                    # tokenizer.save_pretrained(save_dir + "/best_val_auc_tokenizer")
                if eval_acc > best_val_acc:
                    updated.append("ACC")
                    best_val_acc = eval_acc

                if eval_macro_f1 > best_val_macroF1:
                    updated.append("Macro F1")
                    best_val_macroF1 = eval_macro_f1

                if updated:
                    tst_auc, tst_macro_F1, tst_acc, _ = evaluate(tokenizer, model, args.device,
                                                                 tst_dataloader, model_type=args.ptlm)
                    print('Current tst eval: AUC ', tst_auc, 'Best AUC: ', best_tst_auc)
                    print('Current tst eval: ACC ', tst_acc, 'Best ACC: ', best_tst_acc)
                    print('Current tst eval: Macro F1 ', tst_macro_F1, 'Best Tuned_ACC: ', best_tst_macroF1)
                    if tst_acc > best_tst_acc:
                        best_tst_acc = tst_acc
                        torch.save(model.state_dict(), save_dir + f"/best_tst_acc_model_seed_{args.seed}.pth")
                        tokenizer.save_pretrained(save_dir + "/best_tst_acc_tokenizer")
                    if tst_auc > best_tst_auc:
                        best_tst_auc = tst_auc
                        # torch.save(model.state_dict(), save_dir + f"/best_tst_auc_model_seed_{args.seed}.pth")
                        # tokenizer.save_pretrained(save_dir + "/best_tst_auc_tokenizer")
                    if tst_macro_F1 > best_tst_macroF1:
                        best_tst_macroF1 = tst_macro_F1
                        # torch.save(model.state_dict(), save_dir + f"/best_tst_tunedAcc_model_seed_{args.seed}.pth")
                        # tokenizer.save_pretrained(save_dir + "/best_tst_tunedAcc_tokenizer")

                    logger.info(
                        f"Validation {updated} reached best at epoch {e} step {iteration}, evaluating on test set")
                    logger.info(
                        "Best Test Scores: AUC: {}\tACC: {}\t Macro F1: {}\t\n"
                        "Best Dev Scores: AUC: {}\tACC: {}\t Macro F1: {}\t".format(best_tst_auc,
                                                                                    best_tst_acc,
                                                                                    best_tst_macroF1,
                                                                                    best_val_auc,
                                                                                    best_val_acc,
                                                                                    best_tst_macroF1))
                model.train()

    tst_results = [best_tst_acc, best_tst_auc, best_tst_macroF1]
    np.save(os.path.join(save_dir, f"tst_results_{args.ptlm.split('/')[-1]}_{args.seed}.npy"), tst_results)


if __name__ == "__main__":
    main()
