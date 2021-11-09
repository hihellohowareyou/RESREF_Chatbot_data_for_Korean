"""ko-bart로 단순 generator"""
from transformers import BartForConditionalGeneration,trainer
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from dataset import wellnessDataset


def train(args):
    model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--batch_size", type=int, default=22, help="model name")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=10e-5, help="dataset")
    parser.add_argument("--warmup_steps", type=int, default=500, help="dataset")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="dataset")

    args = parser.parse_args()
    train(args)
