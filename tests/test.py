from abs_summarization.constants import Directories
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained(Directories.LLM_DIR.joinpath("t5-base"))
model = T5ForConditionalGeneration.from_pretrained(
    Directories.LLM_DIR.joinpath("t5-base-trained-1-epoch")
)

summarize_prefix = "summarize: "
# query = input("To summarize: ")
with open("sample_text.txt") as file:
    query = file.read()
input_ids = tokenizer(
    summarize_prefix + query, max_length=512, padding="max_length", return_tensors="pt"
).input_ids
outputs = model.generate(input_ids, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""
from:
he was named by time magazine among the 100 most influential people in the world. he has maintained his sobriety since 2003. 
he starred in two teen films, weird science and less than zero.

to:
Robert John Downey Jr. was named by Time magazine among the 100 most influential people in the world in 2008 and from 2013 to 2015, 
Forbes named him Hollywood's highest-paid actor. He made his acting debut in 1970 in his father Robert Downey Sr.'s film Pound. 
He subsequently worked with the Brat Pack in the teen films Weird Science (1985) and Less than Zero (1987).
"""

"""
from:
carbon dioxide is a chemical compound with the chemical formula CO2. 
it is made up of molecules that each have one carbon atom covalently double bonded to two oxygen atoms. 
it is the primary carbon source for life on earth.

to:
carbon dioxide is a chemical compound with the chemical formula CO2. 
It is made up of molecules that each have one carbon atom covalently double bonded to two oxygen atoms. 
Carbon dioxide is the primary carbon source for life on Earth.
"""
