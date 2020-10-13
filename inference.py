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
BATCH_SIZE = 10

def inference():
    SEED = random.randint(0, 9999)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_return_sequences', help='num_return_sequences=1', type=int)
    parser.add_argument('--no_fp16', help='')
    args = parser.parse_args()
    num_return_sequences = args.num_return_sequences
    print("args.no_fp16: ", args.no_fp16)
    if(args.no_fp16 == "True"):
        fp16 = False
    else:
        fp16 = True
    additional_args = ["--model_type="+MODEL_TYPE, "--model_name_or_path="+MODEL_PATH, "--seed=" + str(SEED), "--length=350"]
    if(fp16):
        additional_args.append("--fp16")
    prompt = input("Model prompt >>> ")
    no_batches = math.ceil(num_return_sequences/BATCH_SIZE);
    seqs_list = []
    start_time = time.time()
    for i in range(no_batches):
        print("progress batch no: {} /{}".format(i, no_batches))
        num_to_generate = min(BATCH_SIZE, num_return_sequences-len(seqs_list)*BATCH_SIZE)
        sequences = infer(prompt, additional_args, num_to_generate)
        seqs_list.append(sequences)
    all_seqs = [seq for sequences in seqs_list for seq in sequences]
    print("all_seqs: ", all_seqs)
    print("text generation took --- %s seconds ---" % (time.time() - start_time))
    ts = time.time()
    filename = DATA_PATH+"/inference_results_"+datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M') + ".csv"
    print("saving results to: ", filename)
    with open(filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(all_seqs)

def infer(prompt, args, num_return_sequences):
    args.append("--prompt="+prompt)
    args.append("--num_return_sequences="+str(num_return_sequences))
    return run_generation.main(args)



if __name__ == "__main__":
    inference()