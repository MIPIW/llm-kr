import re
import pandas as pd
from tqdm.auto import tqdm
from unicodedata import name
from multiprocessing import Pool

from evaluate import load
from datasets import load_dataset

# Precompile the bad words regex pattern
badwords_df = pd.read_excel("../badwords_collection_bw_opendict.xlsx")
badwords = sorted(list(badwords_df["entry"]))

wer_metric = load("wer")

def has_honorific(text):
    for i, char in enumerate(text[1:]):
        if char == "니" and name(text[i]).endswith("B"):
            return True
    return False

def has_badwords(text):
    for bw in badwords:
        if bw in text:
            return True
    return False

def edit_distance(args):
    translated, inspected = args
    wer = wer_metric.compute(references=[translated], predictions=[inspected])
    return wer

def is_high_quality(args):
    translated, inspected = args
    if len(inspected) < 100:
        return False
    if re.search(r"[^ -~가-힣]", inspected):
        return False
    if not has_honorific(inspected):
        return False
    if has_badwords(inspected):
        return False
    if edit_distance((translated, inspected)) < 0.1:
        return False
    return True

def main():  
    print("Loading dataset")
    df = pd.read_csv("../datasets/oig-smallchip2-dedu-slice_reviewed_week1-8.csv", keep_default_na=False)
    
    # Prepare arguments for parallel processing
    args = [(df.loc[i]["번역문A"], df.loc[i]["검수A"]) for i in range(len(df))]
    
    # Use multiprocessing to apply is_high_quality in parallel
    with Pool(processes=36) as pool:
        results = list(tqdm(pool.imap(is_high_quality, args), total=len(df), desc="Filtering dataset"))
    
    # Filter the DataFrame based on the results
    df = df[results]
    
    print("Saving filtered dataset")
    df.to_csv("../datasets/oig-smallchip2-dedu-slice_reviewed_week1-8_cleaned.csv", index=False, na_rep="nan")
    
    print("Done")
    
if __name__ == "__main__":
    main()
