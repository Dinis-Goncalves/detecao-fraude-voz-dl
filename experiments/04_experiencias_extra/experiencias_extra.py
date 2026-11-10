# Experiências extra (04). Três testes complementares, escolhidos pelo primeiro argumento:
#
#   python experiencias_extra.py todos-contra-todos
#       Treina em cada dataset (os 4 individuais e o TeleAntiFraud) e testa em todos,
#       para medir a generalização entre datasets. Só usa os 2 melhores modelos.
#
#   python experiencias_extra.py menos-palavras
#       Deteção o mais cedo possível: o modelo treinado é testado com conversas
#       truncadas (5%, 10%, ... 100% das palavras) para simular um cenário de tempo
#       real e perceber a partir de que ponto consegue detetar fraude.
#
#   python experiencias_extra.py tamanhos-teste
#       Influência do tamanho do split: aumenta-se a fatia de teste (20% a 90%) à
#       custa do treino, para ver como a performance cai com menos dados de treino.
#       Usa os 2 datasets mais complexos: o merged e o TeleAntiFraud.
#
# Opção: --epochs N (por omissão 10; usa 1 para um teste rápido).
#
# Labels:
#   1 = scam/fraude
#   0 = não scam/não fraude
#
# Pré-processamento: lowercasing, uniformização das marcas de orador e remoção de
# espaços a mais. Os números NÃO são substituídos por <NUM>.

import argparse
import re
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Raiz do repositório (este ficheiro está em experiments/04_experiencias_extra/)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import dos modelos
from models.base import create_model as create_base
from models.lstm import create_model as create_lstm
from models.gru import create_model as create_gru
from models.bi_lstm import create_model as create_bilstm
from models.bi_gru import create_model as create_bigru
from models.cnn import create_model as create_cnn

# Pastas de dados e de resultados
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INDIVIDUAL_PATHS = {
    "MASC": DATA_DIR / "MASC (Multi Agent Scam Conversation)" / "agent_conversation_all.csv",
    "SSC": DATA_DIR / "SSC (Scammer Conversation)" / "gen_conver_noIdentifier_1000.csv",
    "SSD": DATA_DIR / "SSD (Scam Dialogue)" / "scam-dialogue_all.csv",
    "SASC": DATA_DIR / "SASC (Single Agent Scam Conversation)" / "single-agent-scam-dialogue_all.csv",
}
MERGED_PATH = DATA_DIR / "merged" / "merged.csv"
TRANSCRIPT_PATH = DATA_DIR / "TeleAntiFraud_Call_Transcript" / "transcript.csv"

SEED = 42
VOCAB_SIZE = 10000

# Os 6 modelos, e apenas os 2 melhores (usados no todos-contra-todos)
ALL_MODELS = {
    "Base": create_base,
    "LSTM": create_lstm,
    "GRU": create_gru,
    "Bi-LSTM": create_bilstm,
    "Bi-GRU": create_bigru,
    "CNN": create_cnn,
}
BEST_MODELS = {
    "Bi-LSTM": create_bilstm,
    "CNN": create_cnn,
}

PERCENT_STEPS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]   # menos-palavras
TEST_SIZES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]        # tamanhos-teste


# ------------------------- Dados e pré-processamento ------------------------- #

def clean_text(text, harmonizar_oradores=False):
    """
    Limpeza leve do texto. Com harmonizar_oradores=True uniformiza também as
    marcas de orador dos vários datasets (necessário ao testar entre datasets).
    """
    text = str(text).lower()
    text = text.replace("[left]", "[CALLER]")
    text = text.replace("[right]", "[RECEIVER]")
    text = text.replace("caller:", "[CALLER]")
    text = text.replace("receiver:", "[RECEIVER]")
    if harmonizar_oradores:
        text = text.replace("suspect:", "[CALLER]")
        text = text.replace("innocent:", "[RECEIVER]")
        text = text.replace("person a:", "[CALLER]")
        text = text.replace("person b:", "[RECEIVER]")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_irregular_csv(dataset_path):
    """
    Os diálogos contêm vírgulas internas, por isso o carregamento é manual:
    divide-se a linha a partir da direita e lê-se a label no fim (igual à experiência 01).
    """
    rows = []
    ignoradas = 0

    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
        expected_cols = len(header)

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.rsplit(",", expected_cols - 1)
            if len(parts) != expected_cols:
                ignoradas += 1
                continue

            label = parts[-1].strip().replace('"', "")
            if label not in ["0", "1"]:
                ignoradas += 1
                continue

            rows.append([parts[0].strip(), int(label)])

    if ignoradas:
        print(f"{dataset_path.name}: {ignoradas} linhas ignoradas")

    df = pd.DataFrame(rows, columns=["text", "label"])
    if df.empty:
        raise ValueError(f"O dataset {dataset_path.name} ficou vazio após o carregamento.")
    return df


def load_transcript():
    """TeleAntiFraud sem as instâncias 'unlabeled' e com as labels em 0/1."""
    df = pd.read_csv(TRANSCRIPT_PATH)
    df = df[df["label"] != "unlabeled"].copy()
    df["label"] = df["label"].map({"fraud": 1, "not_fraud": 0})
    return df


def preparar_tokenizer(x_train):
    """Tokenizer ajustado APENAS no treino; max_len = percentil 95 das sequências de treino."""
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)
    seqs = tokenizer.texts_to_sequences(x_train)
    max_len = int(np.percentile([len(s) for s in seqs], 95))
    return tokenizer, max_len


def codificar(tokenizer, texts, max_len):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")


def calcular_metricas(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def prever(model, x_seq):
    return (model.predict(x_seq, verbose=0) > 0.5).astype(int).flatten()


def treinar_com_validacao(creator, x_train_seq, y_train, max_len, epochs):
    """Treino com 15% de validação e EarlyStopping (usado em menos-palavras e tamanhos-teste)."""
    tf.keras.backend.clear_session()
    model = creator(max_len=max_len, vocab_size=VOCAB_SIZE)
    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(
        x_train_seq,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.15,
        verbose=0,
        callbacks=[early_stop]
    )
    return model


def datasets_merged_e_transcript():
    merged = pd.read_csv(MERGED_PATH)
    transcript = load_transcript()
    datasets = {"merged": merged, "transcript": transcript}
    for df in datasets.values():
        df["clean_text"] = df["text"].apply(clean_text)
    return datasets


# ------------------------------ Experiência 1 ------------------------------ #

def plot_heatmap(results_df, nomes, metric, model_name):
    sub = results_df[results_df["Model"] == model_name]
    matrix = sub.pivot(index="Train", columns="Test", values=metric).loc[nomes, nomes]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        cbar=True
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.xlabel("Test dataset")
    plt.ylabel("Train dataset")
    plt.title(f"{model_name} - {metric}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"todos_contra_todos_{model_name}_{metric}.png", dpi=150)


def todos_contra_todos(epochs):
    datasets = {nome: load_irregular_csv(path) for nome, path in INDIVIDUAL_PATHS.items()}
    datasets["TeleAntiFraud"] = load_transcript()
    for df in datasets.values():
        df["clean_text"] = df["text"].apply(lambda t: clean_text(t, harmonizar_oradores=True))

    nomes = list(datasets.keys())
    rows = []

    for train_name, train_df in datasets.items():
        print(f"\nTraining on: {train_name}")

        # 20% do dataset de treino fica de fora para avaliar o próprio dataset.
        # Os restantes datasets são usados por inteiro (nunca foram vistos no treino).
        x_train, x_holdout, y_train, y_holdout = train_test_split(
            train_df["clean_text"].values,
            train_df["label"].values,
            test_size=0.2,
            stratify=train_df["label"].values,
            random_state=SEED
        )

        tokenizer, max_len = preparar_tokenizer(x_train)
        x_train_seq = codificar(tokenizer, x_train, max_len)

        for model_name, creator in BEST_MODELS.items():
            print(f"  Model: {model_name}")

            tf.keras.backend.clear_session()
            model = creator(max_len=max_len, vocab_size=VOCAB_SIZE)
            model.fit(x_train_seq, y_train, epochs=epochs, batch_size=32, verbose=0)

            for test_name, test_df in datasets.items():
                if test_name == train_name:
                    x_eval, y_eval, avaliacao = x_holdout, y_holdout, "holdout (20%)"
                else:
                    x_eval = test_df["clean_text"].values
                    y_eval = test_df["label"].values
                    avaliacao = "dataset completo"

                y_pred = prever(model, codificar(tokenizer, x_eval, max_len))

                rows.append({
                    "Train": train_name,
                    "Model": model_name,
                    "Test": test_name,
                    "avaliacao": avaliacao,
                    "n_test": len(y_eval),
                    **calcular_metricas(y_eval, y_pred)
                })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "todos_contra_todos.csv", index=False)

    for model_name in BEST_MODELS:
        for metric in ("f1", "recall"):
            plot_heatmap(results_df, nomes, metric, model_name)

    plt.show()


# ------------------------------ Experiência 2 ------------------------------ #

def truncar_texto(text, percent):
    """Mantém apenas as primeiras 'percent' palavras do texto (pelo menos uma)."""
    words = text.split()
    cutoff = max(1, int(len(words) * percent))
    return " ".join(words[:cutoff])


def menos_palavras(epochs):
    datasets = datasets_merged_e_transcript()
    rows = []

    for dataset_name, df in datasets.items():
        print(f"\nDataset: {dataset_name}")

        x_train, x_test, y_train, y_test = train_test_split(
            df["clean_text"].values,
            df["label"].values,
            test_size=0.2,
            stratify=df["label"].values,
            random_state=SEED
        )

        tokenizer, max_len = preparar_tokenizer(x_train)
        x_train_seq = codificar(tokenizer, x_train, max_len)

        for model_name, creator in ALL_MODELS.items():
            print(f"  Model: {model_name}")
            model = treinar_com_validacao(creator, x_train_seq, y_train, max_len, epochs)

            # Aqui é que se simula a conversa em tempo real: truncam-se os textos de teste
            for percent in PERCENT_STEPS:
                truncated = [truncar_texto(text, percent) for text in x_test]
                y_pred = prever(model, codificar(tokenizer, truncated, max_len))

                rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "percent": percent,
                    **calcular_metricas(y_test, y_pred)
                })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "menos_palavras.csv", index=False)

    for dataset_name in results_df["dataset"].unique():
        plt.figure()
        df_d = results_df[results_df["dataset"] == dataset_name]

        for model_name in df_d["model"].unique():
            df_m = df_d[df_d["model"] == model_name].sort_values("percent")
            plt.plot(df_m["percent"] * 100, df_m["f1"], marker="o", linewidth=2, label=model_name)

        plt.title(f"Early Detection - {dataset_name}")
        plt.xlabel("Conversation Seen (%)")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid()
        plt.savefig(RESULTS_DIR / f"menos_palavras_{dataset_name}.png", dpi=150)

    plt.show()


# ------------------------------ Experiência 3 ------------------------------ #

def tamanhos_teste(epochs):
    datasets = datasets_merged_e_transcript()
    rows = []

    for dataset_name, df in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")

        x = df["clean_text"].values
        labels = df["label"].values

        for test_size in TEST_SIZES:
            x_train, x_test, y_train, y_test = train_test_split(
                x, labels, test_size=test_size, random_state=SEED, stratify=labels
            )

            # Tokenizer e max_len calculados só com o treino de cada split
            tokenizer, max_len = preparar_tokenizer(x_train)
            x_train_seq = codificar(tokenizer, x_train, max_len)
            x_test_seq = codificar(tokenizer, x_test, max_len)

            for model_name, creator in ALL_MODELS.items():
                print(f"[{dataset_name}] {model_name} | test={test_size:.1f}")
                model = treinar_com_validacao(creator, x_train_seq, y_train, max_len, epochs)
                y_pred = prever(model, x_test_seq)

                rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "test_size": test_size,
                    "train_size": round(1 - test_size, 2),
                    **calcular_metricas(y_test, y_pred)
                })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "tamanhos_teste.csv", index=False)

    print(
        results_df
        .groupby(["dataset", "model", "test_size"])[["accuracy", "f1"]]
        .mean()
    )

    # Declínio do F1 com menos dados de treino
    for dataset_name in results_df["dataset"].unique():
        plt.figure()
        df_d = results_df[results_df["dataset"] == dataset_name]

        for model_name in df_d["model"].unique():
            df_m = df_d[df_d["model"] == model_name].sort_values("test_size")
            plt.plot(df_m["test_size"], df_m["f1"], marker="o", linewidth=2, label=model_name)

        plt.title(f"F1 Collapse Curve - {dataset_name}")
        plt.xlabel("Test Size (data scarcity on training side)")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid()
        plt.savefig(RESULTS_DIR / f"tamanhos_teste_{dataset_name}.png", dpi=150)

    # Heatmap da média dos dois datasets
    pivot = results_df.pivot_table(
        index="model", columns="test_size", values="f1", aggfunc="mean"
    )
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, cmap="viridis")
    plt.title("F1 Collapse Heatmap (média dos 2 datasets)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tamanhos_teste_heatmap.png", dpi=150)

    plt.show()


# --------------------------------- Main --------------------------------- #

EXPERIENCIAS = {
    "todos-contra-todos": todos_contra_todos,
    "menos-palavras": menos_palavras,
    "tamanhos-teste": tamanhos_teste,
}

parser = argparse.ArgumentParser()
parser.add_argument("experiencia", choices=list(EXPERIENCIAS.keys()),
                    help="experiência a correr")
parser.add_argument("--epochs", type=int, default=10,
                    help="número máximo de épocas (por omissão: 10)")
args = parser.parse_args()

# Para maior reprodutibilidade
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

EXPERIENCIAS[args.experiencia](args.epochs)
