import os
import random
import datetime
import time
import run_generation
import argparse
FILE_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL_PATH = FILE_PATH + "/models/gpt2_xl_1ephoc"
DATA_PATH = FILE_PATH + "/data"
MODEL_TYPE = "gpt2"

def inference():
    SEED = random.randint(0, 9999)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_return_sequences', help='num_return_sequences=1')
    parser.add_argument('--no_fp16', help='')
    args = parser.parse_args()
    num_return_sequences = args.num_return_sequences
    print("args.no_fp16: ", args.no_fp16)
    if(args.no_fp16 == "True"):
        fp16 = False
    else:
        fp16 = True
    print("num_return_sequences: ", num_return_sequences)
    print("fp16: ", fp16)
    additional_args = ["--num_return_sequences="+num_return_sequences, "--model_type="+MODEL_TYPE, "--model_name_or_path="+MODEL_PATH, "--seed=" + str(SEED), "--length=350"]
    if(fp16):
        additional_args.append("--fp16")
    #when running this file pass flag --num_return_sequences=500 to tell script to generate 500 examples or any number we want
    ts = time.time()
    run_generation.main(additional_args, DATA_PATH+"/inference_results_"+datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M') + ".csv")



if __name__ == "__main__":
    inference()