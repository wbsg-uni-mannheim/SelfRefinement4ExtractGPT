from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()

client.fine_tuning.jobs.create(
    training_file="file-KFb3BcKJWr6vbwpXteAQRs",
    validation_file="file-WPzzGfX9avDbfQdG32uuCW",
    model="gpt-4o-2024-08-06",
    seed=42
)

#client.fine_tuning.jobs.create(
#    training_file="file-du8vHQGRiSuZqcQvqR8WaXFw",
#    validation_file="file-SvPQ2hXVJcsQhKaaJtNwC0AE",
#    model="gpt-4o-mini-2024-07-18",
#    seed=42
#)