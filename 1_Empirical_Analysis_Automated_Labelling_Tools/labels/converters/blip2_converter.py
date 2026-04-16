import pandas as pd
import re

def parse_blip2_gender(raw_answer: str):
    if not isinstance(raw_answer, str):
        return None

    text = raw_answer.lower().strip()

    if "answer:" in text:
        text = text.split("answer:", 1)[1].strip()
    if re.search(r"\b(female|woman)\b", text):
        return "female"
    if re.search(r"\b(male|man)\b", text):
        return "male"

    return None

# Load raw BLIP2 results
df = pd.read_csv("1_Empirical_Analysis_Automated_Labelling_Tools/labels/results/blip2_results.csv")
df["parsed_gender"] = df["Gender"].apply(parse_blip2_gender)
df["parsed_gender"] = df["parsed_gender"].fillna("unlabelled")
# print(df["parsed_gender"].value_counts())
df.to_csv("cleaned_blip2.csv", index=False)