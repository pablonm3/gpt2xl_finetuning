import os
import random
FILE_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL_PATH = "gpt2"# "PATH_TO_MY_MODEL"
MODEL_TYPE = "gpt2"

def inference():
    SEED = random.randint(0, 999)
    arguments= "--model_type={} --model_name_or_path={} --num_return_sequences=3 --seed={} --length=50".format(MODEL_TYPE, MODEL_PATH, SEED)
    os.system("python " + FILE_PATH +"/run_generation.py " + arguments)



if __name__ == "__main__":
    inference()