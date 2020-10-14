import csv
import os
import random
import datetime
import time
import run_generation
import argparse
import math
FILE_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL_PATH = FILE_PATH + "/models/gpt2_xl_1ephoc"
DATA_PATH = FILE_PATH + "/data"
MODEL_TYPE = "gpt2"


def inference():
    SEED = random.randint(0, 9999)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_return_sequences', help='num_return_sequences=1', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--no_fp16', help='', type=bool, default=False)
    args = parser.parse_args()
    num_return_sequences = args.num_return_sequences
    batch_size = args.batch_size
    if(args.no_fp16):
        fp16 = False
    else:
        fp16 = True
    prompt = input("Model prompt >>> ")
    args = ["--num_return_sequences="+str(num_return_sequences), "--prompt="+prompt, "--model_type="+MODEL_TYPE, "--model_name_or_path="+MODEL_PATH, "--seed=" + str(SEED), "--length=350"]
    if(fp16):
        args.append("--fp16")
    start_time = time.time()
    sequences = run_generation.main(args, batch_size)
    print("text generation took --- %s seconds ---" % (time.time() - start_time))
    ts = time.time()
    filename = DATA_PATH+"/inference_results_"+datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M') + ".csv"
    print("saving results to: ", filename)
    with open(filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(sequences)


if __name__ == "__main__":
    inference()