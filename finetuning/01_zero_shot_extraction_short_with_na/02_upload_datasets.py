from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()

client.files.create(
  file=open("data/finetuning/ae-110k/ae-110k_train_records_short_with_na_prompt.jsonl", "rb"),
  purpose="fine-tune"
)

client.files.create(
  file=open("data/finetuning/ae-110k/ae-110k_val_records_short_with_na_prompt.jsonl", "rb"),
  purpose="fine-tune"
)