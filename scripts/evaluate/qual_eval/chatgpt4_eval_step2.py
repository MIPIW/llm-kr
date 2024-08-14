import sys
import re
from tqdm import tqdm
import pandas as pd


def parse_evaluation_result(evaluation_result, num_models):
    fluency_scores = []
    relevance_scores = []
    accuracy_scores = []
    
    is_md_table = False
    
    # 1. 점수 집계 시 사용할 문자열 범위 설정
    m_summary_start_indices = [(m.start()) for m in re.finditer('[Ss]ummary', evaluation_result)]    
    
    if m_summary_start_indices:
        # 마지막에 등장한 score 요약 부분만 읽어서 처리하기
        start_idx = m_summary_start_indices[-1]
        _evaluation_result = evaluation_result[start_idx:]
        
        if re.search("[0-9]", _evaluation_result): 
            # summary 뒤에 점수 없고 줄글인 경우 -> 전체 문자열 대상 정규식 매칭
            # summary 뒤에 점수 있는 경우 -> summary만 보기
            evaluation_result = _evaluation_result
            if "|" in evaluation_result: # markdown table format
                is_md_table = True
      
        
    # 2. 정규식 패턴 설정 & 매칭 결과 수집
    if is_md_table:
        row_pattern = re.compile(r"\|\s*[A-F]\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*")
        row_matches = row_pattern.findall(evaluation_result)
        
        fluency_matches = [ item[0] for item in row_matches ]
        relevance_matches = [ item[1] for item in row_matches ]
        accuracy_matches= [ item[2] for item in row_matches ]
        
        
    else: # string type
        fluency_pattern = re.compile(r"Fluency:\s*(\d+)")
        relevance_pattern = re.compile(r"Relevance:\s*(\d+)")
        accuracy_pattern = re.compile(r"Accuracy:\s*(\d+)")
    
        fluency_matches = fluency_pattern.findall(evaluation_result)
        relevance_matches = relevance_pattern.findall(evaluation_result)
        accuracy_matches= accuracy_pattern.findall(evaluation_result)
        
    if len(fluency_matches) == num_models and len(relevance_matches) == num_models and len(accuracy_matches) == num_models:
        fluency_scores = [int(score) for score in fluency_matches]
        relevance_scores = [int(score) for score in relevance_matches]
        accuracy_scores= [int(score) for score in accuracy_matches]
        flag = True
    else:
        fluency_scores = [None] * num_models
        relevance_scores = [None] * num_models
        accuracy_scores= [None] * num_models
        flag = False
        
    return fluency_scores, relevance_scores, accuracy_scores, flag


def get_scores_model(df, num_models):
    # 모델별 점수 수합
    scores_model = [  [ list(), list(), list()]  for i in range(num_models) ]  # fluency, relevance, accuracy per model
    cnt_complete, cnt_incomplete = 0, 0
    
    for idx, row in tqdm(df.iterrows()):
        evaluation_result = row['result']
        fluency_scores, relevance_scores, accuracy_scores, flag = parse_evaluation_result(evaluation_result, num_models)
    
        for i, score_model in enumerate(scores_model):
            score_model[0].append( fluency_scores[i] ) # fluency
            score_model[1].append( relevance_scores[i] ) # relevance
            score_model[2].append( accuracy_scores[i] ) # accuracy

        if flag:
            cnt_complete += 1
        else:
            cnt_incomplete += 1
            
    print(f"cnt_complete: {cnt_complete}, cnt_incomplete: {cnt_incomplete}")

    return scores_model


def add_score_col(df, scores, num_models):
    model_codes = "".join( [ chr(i) for i in range(ord("A"), ord("A")+ num_models)] )
    
    for i, model_code in enumerate(model_codes):
        model_score = scores[i]
        df[f"fluency_{model_code}"] = model_score[0]
        df[f"relevance_{model_code}"] = model_score[1]
        df[f"accuracy_{model_code}"] = model_score[2]
        
    return df
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"{sys.argv[0]} fname_input num_models")
        
    fname_input = sys.argv[1]
    num_models = int(sys.argv[2])
    
    # 1. chatgpt 정성평가 출력된 엑셀파일 읽기
    df = pd.read_excel(fname_input)

    # 2. 모델별 점수 수합
    print("collecting scores for each model...")
    scores = get_scores_model(df, num_models)

    # 3. dataframe에 점수 저장
    df = add_score_col(df, scores, num_models)

    # 4. 결과 파일 저장
    df.to_excel(fname_input[:-5]+ "_fin.xlsx", index=False)





