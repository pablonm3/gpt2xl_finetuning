import os
FILE_PATH = os.path.dirname(os.path.realpath(__file__))

TRAIN_FILE = FILE_PATH + "/input/wiki.train.raw"
TEST_FILE = FILE_PATH + "/input/wiki.test.raw"

def finetune():
    # more info: https://github.com/huggingface/transformers/tree/master/examples/language-modeling
    # flags info: https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py
    arguments = "--output_dir=output --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file={}".format(TRAIN_FILE)
    os.system("python " + FILE_PATH +"/run_language_modeling.py " + arguments)


if __name__ == "__main__":
    finetune()