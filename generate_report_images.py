import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

warnings.filterwarnings('ignore')

from src.data.loader import load_batadal, load_skab
from src.pipeline import Pipeline
from src.evaluation.visualization import (
    plot_transition_heatmap, 
    plot_confusion_matrix, 
    plot_path_probabilities
)

OUTPUT_DIR = "report_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print("Loading datasets...")
    bat_df = load_batadal("data/raw/BATADAL/BATADAL_dataset04.csv")
    bat_df.columns = bat_df.columns.str.strip()
    bat_feat = [c for c in bat_df.columns if c not in ["DATETIME", "ATT_FLAG"]]
    bat_data = bat_df[bat_feat].values
    bat_labels = bat_df["ATT_FLAG"].values
    bat_split = int(len(bat_data) * 0.3)
    
    skab_df = load_skab("data/raw/SKAB")
    skab_feat = [c for c in skab_df.columns if c not in ["datetime", "anomaly", "changepoint", "source_file", "source_group"]]
    skab_data = skab_df[skab_feat].values
    skab_labels = skab_df["anomaly"].values
    skab_split = int(len(skab_data) * 0.3)
    
    return (bat_data, bat_labels, bat_split), (skab_data, skab_labels, skab_split)

def generate_base_graphs(bat, skab):
    print("1. Generating Heatmaps and Confusion Matrices...")
    bat_data, bat_labels, bat_split = bat
    skab_data, skab_labels, skab_split = skab
    
    pipe_bat = Pipeline("configs/config.yaml")
    pipe_bat.fit(bat_data[:bat_split])
    res_bat = pipe_bat.predict(bat_data[bat_split:])
    
    matrix_bat, vocab_bat = pipe_bat.builder.get_transition_matrix()
    if len(vocab_bat) > 15:
        vocab_bat = vocab_bat[:15]
        matrix_bat = matrix_bat[:15, :15]
    plot_transition_heatmap(matrix_bat, vocab_bat, title="Transition Probability Heatmap - BATADAL", save_path=os.path.join(OUTPUT_DIR, "heatmap_batadal.png"))
    
    y_slice_bat = bat_labels[bat_split:]
    preds_bat = res_bat["predictions"]
    y_slice_bat = y_slice_bat[-len(preds_bat):]
    y_slice_bat = [1 if str(y).strip() == '1.0' or str(y).strip() == '1' else 0 for y in y_slice_bat]
    cm_bat = confusion_matrix(y_slice_bat, preds_bat, labels=[0, 1])
    plot_confusion_matrix(cm_bat, labels=["Normal", "Anomaly"], title="Confusion Matrix - BATADAL Automata", save_path=os.path.join(OUTPUT_DIR, "cm_batadal.png"))
    
    pipe_skab = Pipeline("configs/config.yaml")
    pipe_skab.fit(skab_data[:skab_split])
    res_skab = pipe_skab.predict(skab_data[skab_split:])
    
    matrix_skab, vocab_skab = pipe_skab.builder.get_transition_matrix()
    if len(vocab_skab) > 15:
        vocab_skab = vocab_skab[:15]
        matrix_skab = matrix_skab[:15, :15]
    plot_transition_heatmap(matrix_skab, vocab_skab, title="Transition Probability Heatmap - SKAB", save_path=os.path.join(OUTPUT_DIR, "heatmap_skab.png"))
    
    y_slice_skab = skab_labels[skab_split:]
    preds_skab = res_skab["predictions"]
    y_slice_skab = y_slice_skab[-len(preds_skab):]
    y_slice_skab = [1 if str(y).strip() == '1.0' or str(y).strip() == '1' else 0 for y in y_slice_skab]
    cm_skab = confusion_matrix(y_slice_skab, preds_skab, labels=[0, 1])
    plot_confusion_matrix(cm_skab, labels=["Normal", "Anomaly"], title="Confusion Matrix - SKAB Automata", save_path=os.path.join(OUTPUT_DIR, "cm_skab.png"))

def plot_curves(bat, skab):
    print("2. Generating ROC and PR Curves...")
    bat_data, bat_labels, bat_split = bat
    skab_data, skab_labels, skab_split = skab
    
    for name, data, labels, split in [("BATADAL", bat_data, bat_labels, bat_split), ("SKAB", skab_data, skab_labels, skab_split)]:
        pipe = Pipeline("configs/config.yaml")
        pipe.fit(data[:split])
        res = pipe.predict(data[split:])
        
        y_scores = [1.0 - (e.get("transition_prob", 1.0) or 0.0) for e in res["explanations"]]
        y_test_slice = labels[split:]
        y_test_slice = y_test_slice[-len(y_scores):]
        y_test_slice = [1 if str(y).strip() == '1.0' or str(y).strip() == '1' else 0 for y in y_test_slice]
        
        fpr, tpr, _ = roc_curve(y_test_slice, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - {name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(OUTPUT_DIR, f"roc_curve_{name.lower()}.png"), dpi=150)
        plt.close()
        
        precision, recall, _ = precision_recall_curve(y_test_slice, y_scores)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='purple', lw=2, label=f'PR AUC = {pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {name} Automata')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(OUTPUT_DIR, f"pr_curve_{name.lower()}.png"), dpi=150)
        plt.close()

def generate_state_diagram(bat, skab):
    print("3. Generating Automata State Diagrams...")
    bat_data, _, bat_split = bat
    skab_data, _, skab_split = skab
    
    for name, data, split in [("BATADAL", bat_data, bat_split), ("SKAB", skab_data, skab_split)]:
        pipe = Pipeline("configs/config.yaml")
        pipe.window_size = 4
        pipe.alphabet_size = 3
        pipe.fit(data[:split])
        
        G = nx.DiGraph()
        for src, targets in pipe.builder.transition_probs.items():
            for dst, prob in targets.items():
                if prob > 0.05:
                    G.add_edge(src, dst, weight=prob)
                    
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=1.5, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='#1f78b4', alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', font_weight="bold")
        
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u,v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, arrowsize=20, edge_color='black', connectionstyle='arc3,rad=0.1')
        
        plt.title(f"Automata State Diagram - {name.lower()}_fold0_explainability.json")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"state_diagram_{name.lower()}.png"), dpi=150)
        plt.close()

def plot_metric(df, dataset_name, metric_col, title, ylabel):
    sub_df = df[df["dataset"] == dataset_name.lower()].copy()
    if len(sub_df) == 0:
        return
    sub_df["params"] = "w=" + sub_df["window_size"].astype(str) + ", a=" + sub_df["alphabet_size"].astype(str)
    
    sub_df = sub_df.sort_values(by=["window_size", "alphabet_size"])
    
    plt.figure(figsize=(12, 4))
    plt.plot(sub_df["params"], sub_df[metric_col], marker='o')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=60, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    metric_safe = metric_col.replace('_', '')
    plt.savefig(os.path.join(OUTPUT_DIR, f"param_sensitivity_{metric_safe}_{dataset_name.lower()}.png"), dpi=150)
    plt.close()

def generate_param_sensitivity():
    print("4. Generating Parameter Sensitivity Graphs...")
    if not os.path.exists("results/automata_param_search.csv"):
        print("Param search CSV not found!")
        return
        
    df = pd.read_csv("results/automata_param_search.csv")
    
    for dataset in ["BATADAL", "SKAB"]:
        plot_metric(df, dataset, "f1_score", f"Automata Parameter Sensitivity, F1-score - {dataset}", "f1_score")
        plot_metric(df, dataset, "state_count", f"Automata Parameter Sensitivity, State Count - {dataset}", "state_count")
        plot_metric(df, dataset, "transition_density", f"Automata Parameter Sensitivity, Transition Density - {dataset}", "transition_density")

def main():
    print("Starting comprehensive graphics generation...")
    bat, skab = load_data()
    generate_base_graphs(bat, skab)
    plot_curves(bat, skab)
    generate_state_diagram(bat, skab)
    generate_param_sensitivity()
    print(f"SUCCESS! All requested graphics generated in '{OUTPUT_DIR}/' folder.")

if __name__ == "__main__":
    main()
