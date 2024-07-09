import openai
import pandas as pd
import re


openai.api_key = ####'api key'

df = pd.read_excel("chatgpt4_0624.xlsx")

def evaluate_generations(prompt, response_a, response_b, response_c, response_d):
    evaluation_prompt = f"""
    Please strictly evaluate the performance of three generative models using qualitative evaluation. Provide the fluency, relevance, and accuracy score of model A, B, C, and D.

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
    
    Deduct fluency points if the model generates output in a different language from the prompt.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict evaluator of the generative models."},
            {"role": "user", "content": evaluation_prompt}
        ]
    )

    return response['choices'][0]['message']['content']

results = []


for idx, row in df.iterrows():
    prompt = row['prompt']
    response_a = row['A']
    response_b = row['B']
    response_c = row['C']
    response_d = row['D']
    evaluation_result = evaluate_generations(prompt, response_a, response_b, response_c, response_d)
    results.append(evaluation_result)


df['result'] = results


df.to_excel("evaluation_results.xlsx", index=False)
