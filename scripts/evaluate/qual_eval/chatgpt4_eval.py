# pip install openai
# openai migrate

import openai
import pandas as pd
import re
import sys, os
from openai import OpenAI
from tqdm import tqdm


def chat_gpt(evaluation_prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {"role": "system", "content": "You are a strict evaluator of the generative models."},
        {"role": "user", "content": evaluation_prompt}
        ]
        )
    
    return response.choices[0].message.content.strip()


def evaluate_generations(prompt, response_a, response_b, response_c, response_d, response_e, response_f):
    evaluation_prompt = f"""
    Please strictly evaluate the performance of six generative models using qualitative evaluation. Provide the fluency, relevance, and accuracy score of model A, B, C, D, E, and F.

    \textbf{{Fluency Rubric:}}
    \begin{{itemize}}
        \item \textbf{{5 points}}: Sentences are fluent without noise.
        \item \textbf{{4 points}}: Sentences contain some noise that does not significantly degrade overall fluency, or relevant noise such as PII (e.g., phone numbers, addresses).
        \item \textbf{{3 points}}: Sentences are generally coherent but contain unnecessary and irrelevant noise.
        \item \textbf{{2 points}}: Severe noise is present (e.g., word repetition, symbols). The output makes sense at the phrase level but not at the sentence level.
        \item \textbf{{1 point}}: The output makes no sense at all.
    \end{{itemize}}

    \textbf{{Relevance Rubric:}}
    \begin{{itemize}}
        \item \textbf{{5 points}}: Sentences fully answer the prompt's question.
        \item \textbf{{4 points}}: Sentences include a direct answer to the prompt's question, though not entirely consistent or coherent.
        \item \textbf{{3 points}}: Sentences provide an indirect and low-quality answer to the prompt's question.
        \item \textbf{{2 points}}: Sentences do not answer the prompt's question but are relevant to the topic.
        \item \textbf{{1 point}}: Sentences are not relevant to the prompt.
    \end{{itemize}}

    \textbf{{Accuracy Rubric:}}
    \begin{{itemize}}
        \item \textbf{{5 points}}: All information and details are completely accurate with no errors or distortions.
        \item \textbf{{4 points}}: Most information is accurate with only minor errors that do not significantly affect understanding.
        \item \textbf{{3 points}}: The main information is accurate, but there are some significant errors.
        \item \textbf{{2 points}}: There are numerous errors, and only some of the information is accurate.
        \item \textbf{{1 point}}: The information is mostly incorrect or misleading, with very few accurate details.
    \end{{itemize}}

    \textbf{{Prompt:}} {prompt}

    \textbf{{A:}} {response_a}
    
    \textbf{{B:}} {response_b}

    \textbf{{C:}} {response_c}

    \textbf{{D:}} {response_d}

    \textbf{{E:}} {response_e}

    \textbf{{F:}} {response_f}

    Deduct fluency points if the model generates output in a different language from the prompt.
    """

    response = chat_gpt(evaluation_prompt)

    return response


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"{sys.argv[0]} fname_input")

    # 1. 모델 inference 수합한 파일 읽기
    fname_input = sys.argv[1]
    df = pd.read_excel(fname_input)

    # 2. chatgpt 정성평가 결과 수합
    results = []
    client = OpenAI( api_key="") # private api key # defaults to os.environ.get("OPENAI_API_KEY")
        
    for idx, row in tqdm(df.iterrows()):
        prompt = row['prompt']
        response_a = row['A']
        response_b = row['B']
        response_c = row['C']
        response_d = row['D']
        response_e = row['E']
        response_f = row['F']

        evaluation_result = evaluate_generations(prompt, response_a, response_b, response_c, response_d, response_e, response_f)
        results.append(evaluation_result)
    
    df['result'] = results
    
    # 3. 결과 파일 저장
    fname_output = fname_input[:-5]+"_eval_result.xlsx"
    df.to_excel(fname_output, index=False)
