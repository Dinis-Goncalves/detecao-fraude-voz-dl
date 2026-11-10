# Executa, num único ficheiro:
#    1. Análise exploratória (EDA) de cada dataset;
#    2. Limpeza, divisão, tokenização e padding dos textos;
#    3. Treino e avaliação dos seis modelos;
#    4. Exportação das métricas e dos gráficos comparativos.
#
# Os datasets têm de existir previamente em data/merged/:
#    - merged.csv
#    - merged_shuffled.csv
#    - merged_half.csv
#
# Labels:
#    1 = scam/fraude
#    0 = não scam/não fraude
#
# Metodologia de preparação (igual à experiência 01):
#    1. split treino/validação/teste ANTES da tokenização;
#    2. tokenizer ajustado APENAS no conjunto de treino;
#    3. max_len definido pelo percentil 95 do tamanho das sequências de treino.
#
# Uso:
#    python treinar_dataset_merged.py
#    python treinar_dataset_merged.py --sanity-check
#    python treinar_dataset_merged.py --epochs 1


import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
# ---------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------

# Este ficheiro deve estar em experiments/02_dataset_merged/.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "merged"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
VOCAB_SIZE = 25000
BATCH_SIZE = 32
DEFAULT_EPOCHS = 15

DATASETS = [
    DATA_DIR / "merged.csv",
    DATA_DIR / "merged_shuffled.csv",
    DATA_DIR / "merged_half.csv",
]

# Import dos modelos
from models.base import create_model as create_base
from models.lstm import create_model as create_lstm
from models.gru import create_model as create_gru
from models.bi_lstm import create_model as create_bilstm
from models.bi_gru import create_model as create_bigru
from models.cnn import create_model as create_cnn


def parse_args():
    parser = argparse.ArgumentParser(
        description="EDA, treino e avaliação dos modelos nos datasets merged."
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="baralha as labels de treino para um teste de controlo.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"número máximo de épocas (por omissão: {DEFAULT_EPOCHS}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Reprodutibilidade e modelos
# ---------------------------------------------------------------------
def set_seed(seed=SEED):
    """Define as seeds usadas pelas bibliotecas."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

model_creators = {
    "Base": create_base,
    "LSTM": create_lstm,
    "GRU": create_gru,
    "Bi-LSTM": create_bilstm,
    "Bi-GRU": create_bigru,
    "CNN": create_cnn,
}


# ---------------------------------------------------------------------
# Carregamento e limpeza dos dados
# ---------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "<NUM>", text)
    return text.strip()


def carregar_dataset(dataset_path):
    """Carrega um CSV merged e valida as colunas necessárias."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {dataset_path}\n"
            "Confirma que os datasets já existem em data/merged/."
        )

    df = pd.read_csv(dataset_path)

    colunas_necessarias = {"text", "label"}
    colunas_em_falta = colunas_necessarias.difference(df.columns)
    if colunas_em_falta:
        raise ValueError(
            f"{dataset_path.name} não contém as colunas necessárias: "
            f"{sorted(colunas_em_falta)}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"O dataset {dataset_path.name} contém labels diferentes de 0 e 1."
        )

    if df.empty:
        raise ValueError(f"O dataset {dataset_path.name} ficou vazio.")

    df["clean_text"] = df["text"].apply(clean_text)
    return df


# ---------------------------------------------------------------------
# Análise exploratória (EDA)
# ---------------------------------------------------------------------

def analisar_dataset(df, dataset_path):
    """
    Analisa a distribuição das classes, qualidade dos dados,
    comprimento dos diálogos e palavras mais frequentes.
    Imprime e guarda o relatório em results/eda_<dataset>.txt.
    """
    linhas = []

    def registar(*partes):
        texto = " ".join(str(parte) for parte in partes)
        print(texto)
        linhas.append(texto)

    registar(f"Dataset: {dataset_path.name}")

    # Balanço das classes
    registar("\nDistribuição das labels:")
    registar(df["label"].value_counts().to_string())
    registar("\nDistribuição relativa das labels:")
    registar(df["label"].value_counts(normalize=True).to_string())

    # Qualidade dos dados
    registar("\nValores nulos por coluna:")
    registar(df.isnull().sum().to_string())
    registar("Textos duplicados:", df["text"].duplicated().sum())

    # Tamanho dos diálogos
    registar("\nNúmero de caracteres:")
    registar(df["text"].str.len().describe().to_string())

    word_count = df["text"].apply(lambda texto: len(texto.split()))
    registar("\nNúmero de palavras:")
    registar(word_count.describe().to_string())

    registar("\nMédia de palavras por classe:")
    registar(df.assign(word_count=word_count).groupby("label")["word_count"].mean().to_string())

    # Palavras mais frequentes
    all_words = " ".join(df["text"].str.lower()).split()
    scam_words = " ".join(df.loc[df["label"] == 1, "text"].str.lower()).split()
    non_scam_words = " ".join(df.loc[df["label"] == 0, "text"].str.lower()).split()

    registar("\nPalavras mais comuns (global):", Counter(all_words).most_common(25))
    registar("\nPalavras mais comuns (fraude):", Counter(scam_words).most_common(25))
    registar(
        "\nPalavras mais comuns (não fraude):",
        Counter(non_scam_words).most_common(25),
    )

    output_path = RESULTS_DIR / f"eda_{dataset_path.stem}.txt"
    output_path.write_text("\n".join(linhas), encoding="utf-8")
    print(f"\nAnálise exploratória guardada em: {output_path}")


# ---------------------------------------------------------------------
# Preparação dos dados para os modelos
# ---------------------------------------------------------------------

def preparar_dados(df, dataset_name, vocab_size=VOCAB_SIZE, seed=SEED):
    """
    Divide os dados antes da tokenização e devolve os conjuntos preparados.

    O tokenizer é ajustado apenas com os textos de treino, evitando leakage.
    O comprimento máximo corresponde ao percentil 95 das sequências de treino.
    """
    textos = df["clean_text"].values
    labels = df["label"].values

    # Split treino/teste e, posteriormente, treino/validação
    X_train_full_text, X_test_text, y_train_full, y_test = train_test_split(
        textos,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_full_text,
        y_train_full,
        test_size=0.15,
        stratify=y_train_full,
        random_state=seed,
    )

    print("\nDistribuição após split:")
    print("Train:", np.unique(y_train, return_counts=True))
    print("Val:", np.unique(y_val, return_counts=True))
    print("Test:", np.unique(y_test, return_counts=True))

    # Tokenização: fit apenas no treino
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    train_sequences = tokenizer.texts_to_sequences(X_train_text)
    val_sequences = tokenizer.texts_to_sequences(X_val_text)
    test_sequences = tokenizer.texts_to_sequences(X_test_text)

    # max_len calculado apenas com base no conjunto de treino
    max_len = max(
        1,
        int(np.percentile([len(seq) for seq in train_sequences], 95)),
    )
    print("max_len definido:", max_len)

    def pad(sequences):
        return pad_sequences(
            sequences,
            maxlen=max_len,
            padding="post",
            truncating="post",
        )

    print(
        f"Train: {len(X_train_text)} | Val: {len(X_val_text)} | "
        f"Test: {len(X_test_text)}"
    )

    return {
        "x_train": pad(train_sequences),
        "x_val": pad(val_sequences),
        "x_test": pad(test_sequences),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "max_len": max_len,
        "vocab_size": vocab_size,
    }


# ---------------------------------------------------------------------
# Treino e avaliação
# ---------------------------------------------------------------------

def treinar_e_avaliar(dados, dataset_name, epochs, sanity_check, rng):
    """Treina os seis modelos e devolve as métricas obtidas."""
    x_train = dados["x_train"]
    x_val = dados["x_val"]
    x_test = dados["x_test"]

    y_train = dados["y_train"]
    y_val = dados["y_val"]
    y_test = dados["y_test"]

    max_len = dados["max_len"]
    vocab_size = dados["vocab_size"]

    # No sanity check, apenas as labels usadas no fit são baralhadas.
    y_fit = rng.permutation(y_train) if sanity_check else y_train

    resultados = {}

    for model_name, creator in model_creators.items():
        print(f"\nTraining {model_name}...")

        # Evita acumulação de grafos/modelos anteriores
        tf.keras.backend.clear_session()

        model = creator(max_len=max_len, vocab_size=vocab_size)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )

        history = model.fit(
            x_train,
            y_fit,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0,
        )

        y_prob = model.predict(x_test, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).ravel()

        resultados[model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "best_val_accuracy": max(history.history["val_accuracy"]),
            "epochs_run": len(history.history["loss"]),
        }

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("Distribuição y_test:", np.unique(y_test, return_counts=True))
        print("Distribuição y_pred:", np.unique(y_pred, return_counts=True))
        print("Matriz de confusão:")
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    return resultados


# ---------------------------------------------------------------------
# Apresentação e exportação dos resultados
# ---------------------------------------------------------------------

def apresentar_resultados(results, sanity_check):
    """Imprime os resultados e guarda CSV e gráfico comparativo."""
    rows = []

    for dataset_name, metrics in results.items():
        for model_name, scores in metrics.items():
            rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                **scores,
            })

    results_df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("RESULTADOS FINAIS" + (" (SANITY CHECK)" if sanity_check else ""))
    print("=" * 80)

    for dataset_name, group in results_df.groupby("Dataset", sort=False):
        print(f"\nPara o Dataset: {dataset_name}")
        for _, row in group.iterrows():
            print(
                f" Usando o modelo: {row['Model']} obteve-se: "
                f"Acc={row['accuracy']:.4f}, "
                f"Prec={row['precision']:.4f}, "
                f"Rec={row['recall']:.4f}, "
                f"F1={row['f1']:.4f}"
            )

    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    plot_df = results_df.melt(
        id_vars=["Dataset", "Model"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        sns.barplot(
            x="Dataset",
            y="Score",
            hue="Model",
            data=plot_df[plot_df["Metric"] == metric],
            ax=ax,
        )
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(model_creators),
        title="Models",
    )

    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    titulo = "Model Performance on Merged Datasets"
    if sanity_check:
        titulo += " (label-shuffle sanity check)"
    plt.suptitle(titulo, fontsize=16)

    sufixo = "_sanity_check" if sanity_check else ""
    grafico_path = RESULTS_DIR / f"metricas_dataset_merged{sufixo}.png"
    csv_path = RESULTS_DIR / f"metricas_dataset_merged{sufixo}.csv"

    fig.savefig(grafico_path, dpi=150, bbox_inches="tight")
    results_df.to_csv(csv_path, index=False)

    print(f"\nMétricas guardadas em: {csv_path}")
    print(f"Gráfico guardado em: {grafico_path}")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs tem de ser pelo menos 1.")

    set_seed()
    rng = np.random.default_rng(SEED)

    print("=" * 80)
    print("EXPERIÊNCIA 02 — DATASETS MERGED")
    print("=" * 80)
    print("Sanity check:", "Ativo" if args.sanity_check else "Inativo")
    print("Épocas máximas:", args.epochs)

    results = {}

    for dataset_path in DATASETS:
        dataset_name = dataset_path.name

        print("\n" + "=" * 80)
        print(f"Processing {dataset_name}...")
        print("=" * 80)

        # Carregamento
        df = carregar_dataset(dataset_path)

        print("\nNúmero de linhas carregadas:", len(df))
        print("\nExemplo de texto carregado:")
        print(df["text"].iloc[0][:1000])
        print("Label:", df["label"].iloc[0])

        # EDA
        print("\n" + "-" * 80)
        print(f"ANÁLISE EXPLORATÓRIA — {dataset_name}")
        print("-" * 80)
        analisar_dataset(df, dataset_path)

        # Preparação e treino
        dados = preparar_dados(
            df,
            dataset_name=dataset_name,
            vocab_size=VOCAB_SIZE,
            seed=SEED,
        )
        results[dataset_name] = treinar_e_avaliar(
            dados=dados,
            dataset_name=dataset_name,
            epochs=args.epochs,
            sanity_check=args.sanity_check,
            rng=rng,
        )

    apresentar_resultados(results, args.sanity_check)


if __name__ == "__main__":
    main()
