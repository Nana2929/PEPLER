import json
import math
import os
from argparse import ArgumentParser

import torch
from torch import nn
from tqdm import tqdm
from transformers import GPT2Tokenizer

from utils import Batchify, DataLoader, ids2tokens, now_time

rating_criterion = nn.MSELoss()


def load_model(model_path, device):
    with open(model_path, "rb") as f:
        model = torch.load(f).to(device)
    return model


def evaluate(data, model, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.0
    rating_loss = 0.0
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, seq, mask = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            seq = seq.to(device)  # (batch_size, seq_len)
            mask = mask.to(device)
            outputs, rating_p = model(user, item, seq, mask)
            t_loss = outputs.loss
            r_loss = rating_criterion(rating_p, rating)

            batch_size = user.size(0)
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample, rating_loss / total_sample


def generate(data, model, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, rating, seq, _ = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            text = seq[:, :1].to(device)  # bos, (batch_size, 1)
            for idx in range(seq.size(1)):
                # produce a word at each step
                if idx == 0:
                    outputs, rating_p = model(user, item, text, None)
                    rating_predict.extend(rating_p.tolist())
                else:
                    outputs, _ = model(user, item, text, None, False)
                last_token = outputs.logits[
                    :, -1, :
                ]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(
                    word_prob, dim=1, keepdim=True
                )  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict, rating_predict


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)
    print(now_time() + "Loading data")
    bos = "<bos>"
    eos = "<eos>"
    pad = "<pad>"
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2", bos_token=bos, eos_token=eos, pad_token=pad
    )
    corpus = DataLoader(args.data_path, args.index_dir, tokenizer, args.words)
    test_data = Batchify(corpus.test, tokenizer, bos, eos, args.batch_size)
    # print number of test_data
    print(
        now_time()
        + "Number of test data: {}".format(test_data.total_step * args.batch_size)
    )

    # Run on test data.
    print("=" * 89)

    test_loss, rating_loss = evaluate(test_data, model, device)
    print(
        now_time()
        + "text ppl {:4.4f} on test | Inferencing".format(math.exp(test_loss))
    )
    print(now_time() + "Generating text")
    idss_predicted, rating_predicted = generate(test_data, model, device)
    tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
    out = []
    for idx, t in tqdm(enumerate(corpus.test)):
        t["predict_text"] = tokens_predict[idx]
        out.append(t)

        if idx == 0:
            print("Example:")
            print("Original:", t["text"])
            print("Predicted:", t["predict_text"])
            print(t)
    print(now_time() + "Saving text")
    with open(args.output_path, "w") as f:
        for t in out:
            f.write(json.dumps(t) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="GPT2 model")
    parser.add_argument(
        "--model_path", type=str, default="model.pt", help="path to the trained model"
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="path to the data"
    )
    parser.add_argument(
        "--index_dir", type=str, default="index", help="path to the index"
    )
    parser.add_argument(
        "--words", type=int, default=20, help="number of words in the vocabulary"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument(
        "--output_path",
        type=str,
        default="generated.jsonl",
        help="path to the output",
    )
    args = parser.parse_args()
    main(args)
