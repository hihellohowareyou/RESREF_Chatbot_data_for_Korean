"""ko-bart로 단순 generator"""
from transformers import AutoTokenizer,Seq2SeqTrainingArguments,PreTrainedTokenizerFast,Seq2SeqTrainer,BartForConditionalGeneration,AutoModelForCausalLM
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
import torch
from datasets import load_metric
from dataset import retrieveAndRefineDateset

def train(args):
    model_name = "hyunwoongko/kobart"
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    # tokenizer = get_kobart_tokenizer()
    dataset = retrieveAndRefineDateset('Chatbot_data/ChatbotData.csv')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    print(tokenizer)
    print(len(dataset))
    x = int(len(dataset) * 0.8)
    # train_set, val_set = torch.utils.data.random_split(dataset, [x,len(dataset)-x])
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",  # output directory
        save_total_limit=2,  # number of total save model.
        save_steps=args.save_steps,  # model saving step.
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        learning_rate=args.learning_rate,  # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        # per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=200,  # log saving step.
        # evaluation_strategy="steps",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        # eval_steps=200,  # evaluation step.
        # load_best_model_at_end=True,
        predict_with_generate= True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
)
    trainer.train()
    trainer.save_model()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--batch_size", type=int, default=20, help="model name")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="dataset")
    parser.add_argument("--save_steps", type=int, default=500, help="dataset")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="dataset")
    parser.add_argument("--warmup_steps", type=int, default=500, help="dataset")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="dataset")
    args = parser.parse_args()
    train(args)
