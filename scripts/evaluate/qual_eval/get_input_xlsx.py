import sys, os
import pandas as pd


# 커맨드 예시: (base) [hmcho@nlplab qual_eval]$ python get_input_xlsx.py qual_eval_moe_240718 qual_eval_moe_240718/chatgpt_240718.xlsx



if len(sys.argv) != 3:
    sys.exit(f"python {sys.argv[0]} dirname_input fname_output")

dirname_input = sys.argv[1]
dirname_input = dirname_input + "/" if not dirname_input.endswith("/") else sys.argv[1]
fname_output = sys.argv[2]
fname_output_mapping = fname_output[:-5]+"_mapping.txt"

print("arguments")
print("\tdirname_input", dirname_input)
print("\tfname_output", fname_output)
print("\tfname_output_mapping", fname_output_mapping)



# 1. inference 결과 엑셀파일 읽어오기
fnames = sorted([fname for fname in os.listdir(dirname_input) if fname.endswith("xlsx")])

mapping_dfs = { fname:k for fname, k in zip(fnames, "ABCDEF")  }
print("mapping:", mapping_dfs)

dfs = [pd.read_excel(dirname_input+fname, skiprows=2) for fname in fnames]


# 2. 모델 inference 결과 수합
responses = pd.concat([ df["response"].rename("ABCDEF"[i]) for i, df in enumerate(dfs)], axis=1)
responses.rename(columns= mapping_dfs)

df_input = pd.concat([dfs[0][["Unnamed: 0", "category", "prompt"]] , responses ], axis=1)

# 3. 결과 파일 저장
df_input.to_excel(fname_output, index=False)

with open(fname_output_mapping, "w", encoding="utf-8") as f:
    print(mapping_dfs, file=f)
