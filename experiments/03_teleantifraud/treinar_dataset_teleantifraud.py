# Experiência 03: dataset TeleAntiFraud (transcrições de chamadas, transcript.csv)
# Labels no CSV: 'fraud', 'not_fraud' e 'unlabeled'.
# As instâncias 'unlabeled' são ignoradas no treino e na avaliação (podem depois ser usadas para previsões).
#
# Este script faz 3 coisas, consoante as opções:
#   python dataset_teleantifraud.py --guardar-modelo       # treina os 6 modelos e guarda o CNN
#   python dataset_teleantifraud.py --eda                  # análise exploratória
#   python dataset_teleantifraud.py --inferir-unlabeled    # prevê as labels das instâncias sem label
# Outras opções: --remover-duplicados (treina sem textos duplicados) e
# --epochs N (por omissão 15; usa 1 para um teste rápido, sem --guardar-modelo).
#
# O CNN foi o modelo com melhor performance, por isso é o único guardado em
# saved_models/teleantifraud/ (modelo .keras, tokenizer em JSON e max_len), para
# ser reutilizado na inferência com exatamente o mesmo pré-processamento.
#
# Pré-processamento: lowercasing, remoção de espaços a mais e substituição das
# marcas de orador [left]/[right] por [CALLER]/[RECEIVER].

import argparse
import json
import re
import sys
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Raiz do repositório (este ficheiro está em experiments/03_teleantifraud/)
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
DATASET_PATH = REPO_ROOT / "data" / "TeleAntiFraud_Call_Transcript" / "transcript.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SAVED_DIR = REPO_ROOT / "saved_models" / "teleantifraud"

SEED = 42
VOCAB_SIZE = 10000

# Funções responsáveis pela criação dos modelos
model_creators = {
    "Base": create_base,
    "LSTM": create_lstm,
    "GRU": create_gru,
    "Bi-LSTM": create_bilstm,
    "Bi-GRU": create_bigru,
    "CNN": create_cnn,
}


def clean_text(text):
    text = str(text).lower()
    text = text.replace("[left]", "[CALLER]")
    text = text.replace("[right]", "[RECEIVER]")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def analise_exploratoria():
    """EDA: balanço das classes, nulos/duplicados, tamanho dos textos e palavras comuns."""
    transcript = pd.read_csv(DATASET_PATH)
    linhas = []

    def registar(*partes):
        texto = " ".join(str(p) for p in partes)
        print(texto)
        linhas.append(texto)

    registar("Distribuição das labels:")
    registar(transcript["label"].value_counts())

    registar("\nNúmero de caracteres:")
    registar(transcript["text"].str.len().describe())

    registar("\nNulos por coluna:")
    registar(transcript.isnull().sum())
    registar("Duplicados:", transcript.duplicated().sum())

    # Número de palavras por transcrição
    transcript["word_count"] = transcript["text"].apply(lambda x: len(x.split()))
    registar("\nNúmero de palavras:")
    registar(transcript["word_count"].describe())

    # É expectável que a fraude tenha mais texto
    registar("\nMédia de palavras por classe:")
    registar(transcript.groupby("label")["word_count"].mean())

    # Palavras mais comuns (global e por classe)
    all_words = " ".join(transcript["text"].str.lower()).split()
    fraud_words = " ".join(
        transcript[transcript["label"] == "fraud"]["text"].str.lower()
    ).split()
    not_fraud_words = " ".join(
        transcript[transcript["label"] == "not_fraud"]["text"].str.lower()
    ).split()

    registar("\nPalavras mais comuns (global):", Counter(all_words).most_common(25))
    registar("\nPalavras mais comuns (fraude):", Counter(fraud_words).most_common(25))
    registar("\nPalavras mais comuns (não fraude):", Counter(not_fraud_words).most_common(25))

    output_path = RESULTS_DIR / "eda_teleantifraud.txt"
    output_path.write_text("\n".join(linhas), encoding="utf-8")
    print(f"\nGuardado em {output_path}")


def inferir_unlabeled():
    """Usa o CNN guardado para prever as labels das instâncias 'unlabeled'."""
    model_path = SAVED_DIR / "cnn_model.keras"
    tokenizer_path = SAVED_DIR / "tokenizer.json"
    config_path = SAVED_DIR / "config.json"

    for path in (model_path, tokenizer_path, config_path):
        if not path.exists():
            raise SystemExit(f"Falta {path}. Corre primeiro o treino com --guardar-modelo.")

    cnn_model = load_model(model_path)
    tokenizer = tokenizer_from_json(tokenizer_path.read_text(encoding="utf-8"))
    max_len = json.loads(config_path.read_text(encoding="utf-8"))["max_len"]

    # Ir buscar as instâncias unlabeled
    transcript = pd.read_csv(DATASET_PATH)
    unlabeled = transcript[transcript["label"] == "unlabeled"].copy()

    # Exatamente o mesmo pré-processamento, tokenizer e padding usados no treino
    unlabeled["clean_text"] = unlabeled["text"].apply(clean_text)
    sequences = tokenizer.texts_to_sequences(unlabeled["clean_text"])
    x_unlabeled = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    # Previsões
    probs = cnn_model.predict(x_unlabeled, verbose=0).flatten()
    unlabeled["fraud_probability"] = probs
    unlabeled["predicted_label"] = (probs > 0.5).astype(int)

    n_fraud = int(unlabeled["predicted_label"].sum())
    n_not_fraud = len(unlabeled) - n_fraud
    resumo = (
        f"Instâncias unlabeled: {len(unlabeled)}\n"
        f"Previstas como fraude: {n_fraud}\n"
        f"Previstas como não fraude: {n_not_fraud}\n"
    )
    print(resumo)

    # Guarda apenas o id e as previsões (sem o texto das transcrições)
    unlabeled[["id", "fraud_probability", "predicted_label"]].to_csv(
        RESULTS_DIR / "inferencias_unlabeled.csv", index=False
    )
    (RESULTS_DIR / "inferencias_unlabeled_resumo.txt").write_text(resumo, encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--eda", action="store_true",
                    help="corre a análise exploratória e termina")
parser.add_argument("--inferir-unlabeled", action="store_true",
                    help="prevê as instâncias sem label com o CNN guardado e termina")
parser.add_argument("--guardar-modelo", action="store_true",
                    help="no treino, guarda o CNN, o tokenizer e o max_len em saved_models/")
parser.add_argument("--remover-duplicados", action="store_true",
                    help="remove textos duplicados antes do split")
parser.add_argument("--epochs", type=int, default=15,
                    help="número de épocas (por omissão: 15)")
args = parser.parse_args()

if args.eda:
    analise_exploratoria()
    raise SystemExit

if args.inferir_unlabeled:
    inferir_unlabeled()
    raise SystemExit


# ------------------------------ TREINO ------------------------------ #

# Para maior reprodutibilidade
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Load do dataset
transcript = pd.read_csv(DATASET_PATH)

# Exemplos sem label ficam de fora da fase de treino/teste
n_unlabeled = (transcript["label"] == "unlabeled").sum()
transcript = transcript[transcript["label"] != "unlabeled"].copy()
print(f"Exemplos 'unlabeled' ignorados: {n_unlabeled}")

# Mapear labels para valores numéricos
transcript["label"] = transcript["label"].map({
    "fraud": 1,
    "not_fraud": 0
})

# Remoção opcional de duplicados (evita o mesmo texto em treino e teste)
if args.remover_duplicados:
    antes = len(transcript)
    transcript = transcript.drop_duplicates().copy()
    print(f"Duplicados removidos: {antes - len(transcript)}")

# Limpar texto
transcript["clean_text"] = transcript["text"].apply(clean_text)

# Separação entre texto e labels
X_text = transcript["clean_text"].values
labels = transcript["label"].values

print("\nDistribuição das labels:")
print(transcript["label"].value_counts())
print(transcript["label"].value_counts(normalize=True))

# Split dos dados em treino, teste e validação
# O split é feito ANTES da tokenização para evitar fuga de informação.
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


# Tokenização
# O tokenizador é treinado apenas com os textos do conjunto de treino.
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

train_sequences = tokenizer.texts_to_sequences(X_train_text)
val_sequences = tokenizer.texts_to_sequences(X_val_text)
test_sequences = tokenizer.texts_to_sequences(X_test_text)


# Padding
# max_len calculado apenas com base nas sequências de treino.
max_len = int(np.percentile([len(seq) for seq in train_sequences], 95))

print("\nmax_len:", max_len)

x_train = pad_sequences(train_sequences, maxlen=max_len, padding="post", truncating="post")
x_val = pad_sequences(val_sequences, maxlen=max_len, padding="post", truncating="post")
x_test = pad_sequences(test_sequences, maxlen=max_len, padding="post", truncating="post")


# Lista a ser usada para guardar as métricas
rows = []

# Loop de cada modelo
for model_name, creator in model_creators.items():
    print(f"\nTraining {model_name}...")

    # Limpa sessão para evitar acumulação de grafos/modelos anteriores
    tf.keras.backend.clear_session()

    model = creator(max_len=max_len, vocab_size=VOCAB_SIZE)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=32,
        verbose=0
    )

    y_prob = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    rows.append({
        "Model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "best_val_accuracy": max(history.history["val_accuracy"]),
    })

    # Guarda o CNN (melhor modelo) e o que é preciso para o reutilizar na inferência
    if args.guardar_modelo and model_name == "CNN":
        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        model.save(SAVED_DIR / "cnn_model.keras")
        (SAVED_DIR / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
        (SAVED_DIR / "config.json").write_text(
            json.dumps({"max_len": max_len, "vocab_size": VOCAB_SIZE, "seed": SEED}, indent=2),
            encoding="utf-8"
        )

    print("Distribuição y_pred:", np.unique(y_pred, return_counts=True))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


results_df = pd.DataFrame(rows)


# Print dos resultados de cada modelo
print("\n" + "=" * 80)
print("RESULTADOS FINAIS")
print("=" * 80)

for _, r in results_df.iterrows():
    print(
        f" Usando o modelo: {r['Model']} obteve-se: "
        f"Acc={r['accuracy']:.4f}, "
        f"Prec={r['precision']:.4f}, "
        f"Rec={r['recall']:.4f}, "
        f"F1={r['f1']:.4f}, "
        f"ROC={r['roc_auc']:.4f}"
    )


# Visualizar
metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
plot_df = results_df.set_index("Model")[metrics_to_plot]

ax = plot_df.plot(kind="bar", figsize=(10, 6))

# Eixo y truncado para distinguir modelos com scores muito próximos
lower = max(0.0, np.floor(plot_df.values.min() * 100) / 100 - 0.01)
ax.set_ylim(lower, 1.0)
ax.set_title("Model Comparison - TeleAntiFraud")
ax.set_ylabel("Score (eixo y truncado)")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()

# Guarda os resultados antes de mostrar o gráfico
fig = ax.get_figure()
fig.savefig(RESULTS_DIR / "metricas_teleantifraud.png", dpi=150)
results_df.to_csv(RESULTS_DIR / "metricas_teleantifraud.csv", index=False)

plt.show()
