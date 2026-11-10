# Cria 2 datasets de teste a partir do dataset merged:
#   - merged_shuffled.csv: as palavras de cada texto são baralhadas
#       -> verificar se os modelos dependem apenas de keywords (sem ordem)
#   - merged_half.csv: cada texto fica com 50% das palavras, escolhidas ao acaso
#       -> verificar se é possível detetar fraude com menos informação
#
# NOTA: random.sample devolve as palavras por ordem aleatória, por isso o
# merged_half.csv também perde a ordem original das palavras.
#
# Por segurança, não sobrescreve ficheiros existentes (usa --overwrite para
# forçar). Os ficheiros gerados com outra seed têm amostras diferentes.

import argparse
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "merged"

SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action="store_true",
                    help="sobrescreve os ficheiros se já existirem")
args = parser.parse_args()

rng = random.Random(SEED)

source_path = DATA_DIR / "merged.csv"
shuffled_path = DATA_DIR / "merged_shuffled.csv"
half_path = DATA_DIR / "merged_half.csv"

for path in (shuffled_path, half_path):
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path.name} já existe. Usa --overwrite para o substituir.")

df = pd.read_csv(source_path)


def shuffle_text(text):
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def half_text(text):
    words = text.split()
    if len(words) <= 1:
        return text
    k = len(words) // 2
    return " ".join(rng.sample(words, k))


df_shuffled = df.copy()
df_shuffled["text"] = df_shuffled["text"].apply(shuffle_text)
df_shuffled.to_csv(shuffled_path, index=False)

df_half = df.copy()
df_half["text"] = df_half["text"].apply(half_text)
df_half.to_csv(half_path, index=False)

print(f"Criados: {shuffled_path.name}, {half_path.name}")
