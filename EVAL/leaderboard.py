import streamlit as st
import os
import json
import pandas as pd

# Set layout to wide mode
st.set_page_config(layout="wide")


def app():
    # 결과를 저장할 데이터프레임을 생성
    data = {}
    avg_data = {}  # average score를 저장하기 위한 dictionary
    # tasks = ["Ko-StrategyQA", "AutoRAGRetrieval", "PublicHealthQA"]
    tasks = [
        "Ko-StrategyQA",
        "AutoRAGRetrieval",
        "MIRACLRetrieval",
        "PublicHealthQA",
        "BelebeleRetrieval",
        "MrTidyRetrieval",
        "MultiLongDocRetrieval",
        # "XPQARetrieval",
        # "KoFinMarketReportRetrieval",
        # "KoFSSFinDictRetrieval",
        # "KoTATQARetrieval",
        # "KoSquadv1Retrieval",
    ]
    top_k_types = ["top1", "top3", "top5", "top10"]

    score_types = {
        "top1": ["recall_at_1", "precision_at_1", "ndcg_at_1"],
        "top3": ["recall_at_3", "precision_at_3", "ndcg_at_3"],
        "top5": ["recall_at_5", "precision_at_5", "ndcg_at_5"],
        "top10": ["recall_at_10", "precision_at_10", "ndcg_at_10"],
    }

    # 각 작업에 대한 데이터를 초기화
    for task in tasks:
        data[task] = {top_k: [] for top_k in top_k_types}

    root_dir = "./RESULTS"

    # 데이터가 저장되어 있는 디렉토리의 모든 하위 폴더를 순회하면서 json 파일을 읽습니다.
    for subdir, dirs, files in os.walk(root_dir):
        if "data_x/EMBEDDING/MODELS/bge-ft-loss=gist-bs=8192-ep=1-lr=3e-5-241016" in os.path.relpath(subdir, root_dir):
            continue
        for file in files:
            for task in tasks:
                if file == task + ".json":
                    with open(os.path.join(subdir, file)) as f:
                        d = json.load(f)

                        for top_k in top_k_types:
                            results = {}

                            for score in score_types[top_k]:
                                if "dev" in d["scores"] and "test" not in d["scores"]:
                                    results[score] = d["scores"]["dev"][0][score]
                                elif "test" in d["scores"] and "dev" not in d["scores"]:
                                    results[score] = d["scores"]["test"][0][score]
                                elif "train" in d["scores"] and "test" not in d["scores"] and "dev" not in d["scores"]:
                                    results[score] = d["scores"]["train"][0][score]
                                else:
                                    results[score] = (d["scores"]["dev"][0][score] + d["scores"]["test"][0][score]) / 2

                            f1_score = (
                                2 * (results[score_types[top_k][1]] * results[score_types[top_k][0]]) / (results[score_types[top_k][1]]+ results[score_types[top_k][0]])
                                if (results[score_types[top_k][1]]+ results[score_types[top_k][0]])> 0
                                else 0
                            )
                            data[task][top_k].append(
                                (
                                    os.path.relpath(subdir, root_dir),
                                    results[score_types[top_k][0]],
                                    results[score_types[top_k][1]],
                                    results[score_types[top_k][2]],
                                    f1_score,
                                )
                            )

    # 각 작업에 대해 top1, top3, top5 점수 표시
    for task in tasks:
        st.markdown(f"# {task}")
        for top_k in top_k_types:
            st.markdown(f"## {top_k.capitalize()} Scores")
            df = pd.DataFrame(
                data[task][top_k],
                columns=[
                    "Subdir",
                    f"Recall_{top_k}",
                    f"Precision_{top_k}",
                    f"NDCG_{top_k}",
                    f"F1_{top_k}",
                ],
            )
            df = df.sort_values(by=f"NDCG_{top_k}", ascending=False)
            st.dataframe(df, use_container_width=True)

            # 각 모델의 평균 점수를 계산
            for subdir, recall, precision, ndcg, f1 in data[task][top_k]:
                if subdir not in avg_data:
                    avg_data[subdir] = {
                        k: [[], [], [], []] for k in top_k_types
                    }  # 각 top_k에 대해 별도 리스트 생성
                avg_data[subdir][top_k][0].append(recall)
                avg_data[subdir][top_k][1].append(precision)
                avg_data[subdir][top_k][2].append(ndcg)
                avg_data[subdir][top_k][3].append(f1)

    # 각 모델 별로 평균 점수를 계산하고 출력합니다.
    st.markdown("# Average Scores")
    for top_k in top_k_types:
        avg_results = []
        for model in avg_data:
            recall_avg = (
                sum(avg_data[model][top_k][0]) / len(avg_data[model][top_k][0])
                if avg_data[model][top_k][0]
                else 0
            )
            precision_avg = (
                sum(avg_data[model][top_k][1]) / len(avg_data[model][top_k][1])
                if avg_data[model][top_k][1]
                else 0
            )
            ndcg_avg = (
                sum(avg_data[model][top_k][2]) / len(avg_data[model][top_k][2])
                if avg_data[model][top_k][2]
                else 0
            )
            f1_avg = (
                sum(avg_data[model][top_k][3]) / len(avg_data[model][top_k][3])
                if avg_data[model][top_k][3]
                else 0
            )
            avg_results.append([model, recall_avg, precision_avg, ndcg_avg, f1_avg])

        avg_df = pd.DataFrame(
            avg_results,
            columns=[
                "Model",
                f"Average Recall_{top_k}",
                f"Average Precision_{top_k}",
                f"Average NDCG_{top_k}",
                f"Average F1_{top_k}",
            ],
        )
        avg_df = avg_df.sort_values(by=f"Average NDCG_{top_k}", ascending=False)
        st.markdown(f"## {top_k.capitalize()} Average Scores")
        st.dataframe(avg_df, use_container_width=True)


if __name__ == "__main__":
    app()