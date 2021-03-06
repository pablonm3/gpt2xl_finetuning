import os
import run_language_modeling

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

TRAIN_FILE = FILE_PATH + "/input/training.txt"
TEST_FILE = FILE_PATH + "/input/wiki.test.raw"

# gpt2 is the 117M version, gpt2-xl is the 1.5B version
MODEL_PATH =  "gpt2-xl"#"gpt2"
MODEL_TYPE = "gpt2"
GRADIENT_CHECKPOINTING = True

# IF WANTED TO IMPROVE RESULTS: increase per_device_train_batch_size and num_train_epochs, the firt one will require higher memory GPUs and the last more time


def finetune():
    # more info: https://github.com/huggingface/transformers/tree/master/examples/language-modeling
    # flags info: https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py
    additional_args = ["--output_dir=" + FILE_PATH + "/output", "--overwrite_output_dir", "--model_type=" + MODEL_TYPE,
                       "--model_name_or_path=" + MODEL_PATH, "--do_train", "--train_data_file=" + TRAIN_FILE,
                       "--per_device_train_batch_size=1", "--num_train_epochs=1", "--fp16"]
    run_language_modeling.main(additional_args, GRADIENT_CHECKPOINTING)


if __name__ == "__main__":
    finetune()
