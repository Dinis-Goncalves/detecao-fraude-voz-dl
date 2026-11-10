# Treino e avaliação dos 6 modelos em cada um dos 4 datasets sintéticos individuais
# (primeira fase do estudo, antes da criação do dataset merged).
# Labels:
#   1 = scam/fraude
#   0 = não scam/não fraude
#
# Alguns datasets contêm colunas adicionais, como "type" e "personality".
# Como os diálogos contêm vírgulas internas, o carregamento é feito manualmente,
# reconstruindo o campo textual a partir da esquerda e lendo a label no fim da linha.

import re
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Raiz do repositório (este ficheiro está em experiments/01_datasets_individuais/)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import dos modelos
from models.base import create_model as create_base
from models.lstm import create_model as create_lstm
from models.bi_lstm import create_model as create_bilstm
from models.gru import create_model as create_gru
from models.bi_gru import create_model as create_bigru
from models.cnn import create_model as create_cnn

# Pastas de dados e de resultados
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Para maior reprodutibilidade
SEED = 42 # para quem não sabe, 42 é a resposta para a vida, o universo e tudo mais ;) Pesquisem a piada!
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# Datasets a serem analisados
datasets = [
    DATA_DIR / "MASC (Multi Agent Scam Conversation)/agent_conversation_all.csv",
    DATA_DIR / "SSC (Scammer Conversation)/gen_conver_noIdentifier_1000.csv",
    DATA_DIR / "SSD (Scam Dialogue)/scam-dialogue_all.csv",
    DATA_DIR / "SASC (Single Agent Scam Conversation)/single-agent-scam-dialogue_all.csv"
]

# Funções responsáveis pela criação dos modelos
model_creators = {
    "Base": create_base,
    "LSTM": create_lstm,
    "GRU": create_gru,
    "Bi-LSTM": create_bilstm,
    "Bi-GRU": create_bigru,
    "CNN": create_cnn
}


# Dicionário a ser usado para guardar as métricas
results = {}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "<NUM>", text)
    return text.strip()


def load_irregular_csv(dataset_path):
    rows = []

    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        header_line = f.readline().strip()
        header = header_line.split(",")

        expected_cols = len(header)

        print(f"\nHeader detetado em {dataset_path.name}: {header}")
        print(f"Número esperado de colunas: {expected_cols}")

        for line_number, line in enumerate(f, start=2):
            line = line.strip()

            if not line:
                continue

            # Divide a partir da direita.
            # Exemplo:
            # text,personality,type,labels -> rsplit(",", 3)
            # text,type,label -> rsplit(",", 2)
            # text,label -> rsplit(",", 1)
            parts = line.rsplit(",", expected_cols - 1)

            if len(parts) != expected_cols:
                print(f"Linha ignorada {line_number}: número inesperado de colunas ({len(parts)})")
                continue

            text = parts[0].strip()
            label = parts[-1].strip().replace('"', "")

            if label not in ["0", "1"]:
                print(f"Linha ignorada {line_number}: label inválida -> {label}")
                continue

            rows.append([text, int(label)])

    df = pd.DataFrame(rows, columns=["text", "label"])

    if df.empty:
        raise ValueError(f"O dataset {dataset_path.name} ficou vazio após o carregamento.")

    df["clean_text"] = df["text"].apply(clean_text)

    return df


# Main loop
for dataset_path in datasets:
    dataset_name = dataset_path.name

    print("\n" + "=" * 80)
    print(f"Processing {dataset_name}...")
    print("=" * 80)

    df = load_irregular_csv(dataset_path)

    print("\nNúmero de linhas carregadas:", len(df))

    print("\nDistribuição das labels:")
    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True))

    print("\nExemplo de texto carregado:")
    print(df["text"].iloc[0][:1000])
    print("Label:", df["label"].iloc[0])

    # Dados textuais e labels
    X_text = df["clean_text"].values
    labels = df["label"].values

    # Split train/test/validation feito ANTES da tokenização
    X_train_full_text, X_test_text, y_train_full, y_test = train_test_split(
        X_text,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_full_text,
        y_train_full,
        test_size=0.15,
        stratify=y_train_full,
        random_state=SEED
    )

    print("\nDistribuição após split:")
    print("Train:", np.unique(y_train, return_counts=True))
    print("Val:", np.unique(y_val, return_counts=True))
    print("Test:", np.unique(y_test, return_counts=True))

    # Tokenização: fit APENAS no treino
    vocab_size = 10000
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(X_train_text)

    train_sequences = tokenizer.texts_to_sequences(X_train_text)
    val_sequences = tokenizer.texts_to_sequences(X_val_text)
    test_sequences = tokenizer.texts_to_sequences(X_test_text)

    # max_len calculado APENAS com base no conjunto de treino
    max_len = int(np.percentile([len(seq) for seq in train_sequences], 95))

    print("max_len definido:", max_len)

    x_train = pad_sequences(
        train_sequences,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    x_val = pad_sequences(
        val_sequences,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    x_test = pad_sequences(
        test_sequences,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    results[dataset_name] = {}

    # Loop de cada modelo
    for model_name, creator in model_creators.items():
        print(f"\nTraining {model_name}...")

        # Limpa sessão para evitar acumulação de grafos/modelos anteriores
        tf.keras.backend.clear_session()

        model = creator(max_len=max_len, vocab_size=vocab_size)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=15,
            batch_size=32,
            verbose=0
        )

        y_prob = model.predict(x_test, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[dataset_name][model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")

        print("Distribuição y_test:", np.unique(y_test, return_counts=True))
        print("Distribuição y_pred:", np.unique(y_pred, return_counts=True))

        print("Matriz de confusão:")
        print(confusion_matrix(y_test, y_pred))


# Print dos resultados de cada modelo para cada dataset
print("\n" + "=" * 80)
print("RESULTADOS FINAIS")
print("=" * 80)

for dataset, metrics in results.items():
    print(f"\nPara o Dataset: {dataset}")
    for model_name, score in metrics.items():
        print(
            f" Usando o modelo: {model_name} obteve-se: "
            f"Acc={score['accuracy']:.4f}, "
            f"Prec={score['precision']:.4f}, "
            f"Rec={score['recall']:.4f}, "
            f"F1={score['f1']:.4f}"
        )


# Plot da evolução das métricas
metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
model_names = list(model_creators.keys())

plot_data = []

for dataset_path in datasets:
    dataset_name = dataset_path.name

    for model in model_names:
        for metric in metrics_to_plot:
            plot_data.append({
                "Dataset": dataset_name,
                "Model": model,
                "Metric": metric,
                "Score": results[dataset_name][model][metric]
            })

plot_df = pd.DataFrame(plot_data)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]

    sns.barplot(
        x="Dataset",
        y="Score",
        hue="Model",
        data=plot_df[plot_df["Metric"] == metric],
        ax=ax
    )

    ax.set_title(metric.capitalize())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)

handles, labels_legend = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels_legend,
    loc="upper center",
    ncol=len(model_names),
    title="Models"
)

for ax in axes:
    ax.get_legend().remove()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle("Model Performance Across Datasets", fontsize=16)

# Guarda os resultados antes de mostrar o gráfico
fig.savefig(RESULTS_DIR / "metricas_datasets_individuais.png", dpi=150)
plot_df.to_csv(RESULTS_DIR / "metricas_datasets_individuais.csv", index=False)

plt.show()
