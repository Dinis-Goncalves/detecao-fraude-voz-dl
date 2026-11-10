# 📞 Deteção de Fraude em Comunicações de Voz usando Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Este repositório contém a versão reestruturada e otimizada do projeto de licenciatura desenvolvido no **Instituto Superior de Engenharia de Coimbra (ISEC)**.

O objetivo do projeto é avaliar e comparar a capacidade de diferentes arquiteturas de **Deep Learning** (modelos de pooling, RNNs unidirecionais/bidirecionais e CNNs 1D) na identificação automática de chamadas de voz fraudulentas (*vishing*, *phishing* telefónico e engenharia social) a partir de transcrições de áudio.

---

## 📌 Visão Geral & Engenharia de Dados

A análise de fraude baseada em transcrições automáticas de voz (ASR) apresenta desafios únicos: variação de sintaxe entre oradores, presença de ruído textual, inconsistência em identificadores de turnos e risco de *data leakage* durante o pré-processamento.

Nesta implementação refatorada, o fluxo de dados foi construído com rigor metodológico:

### 🛠️ Destaques da Implementação e Pré-Processamento
- **Parsing Robusto de CSVs Irregulares (`rsplit`):** Diálogos conversacionais contêm frequentemente vírgulas no texto livre. Para evitar que quebras sintáticas corrompam os ficheiros, a leitura é feita reconstruindo a linha a partir da direita (`rsplit(',', expected_cols - 1)`), isolando a *label* com precisão.
- **Uniformização de Oradores (`[CALLER]` / `[RECEIVER]`):** Unificação de anotações heterogéneas dos datasets (`suspect:`, `innocent:`, `person a:`, `person b:`, `caller:`, `receiver:`, `[left]`, `[right]`) para tokens normalizados, preservando a semântica de quem fala sem viciar a aprendizagem do modelo.
- **Zero-Leakage Tokenization & Padding Dinâmico:** O `Tokenizer` é ajustado (*fit*) **estritamente** no conjunto de treino. O comprimento máximo de sequência (`max_len`) é calculado dinamicamente com base no **percentil 95** das sequências de treino, evitando desperdício de memória e truncamento excessivo.
- **Normalização Textual:** *Lowercasing*, remoção de espaços em excesso e substituição opcional de numerais pelo token `<NUM>`.

---

## 🏗️ Arquiteturas de Deep Learning Implementadas

Todas as arquiteturas foram desenvolvidas em **TensorFlow / Keras**, compiladas com `loss="binary_crossentropy"`, otimizador `Adam` e camada final `Dense(1, activation="sigmoid")`:

| Modelo | Estrutura da Rede | Função Principal |
| :--- | :--- | :--- |
| **Base** | `Embedding(128d)` $\rightarrow$ `GlobalAveragePooling1D` $\rightarrow$ `Dense(64, ReLU)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Output` | *Baseline* leve para verificar separabilidade linear/média. |
| **LSTM** | `Embedding` $\rightarrow$ `SpatialDropout1D(0.2)` $\rightarrow$ `LSTM(64)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Output` | Captura de dependências sequenciais temporais. |
| **GRU** | `Embedding` $\rightarrow$ `SpatialDropout1D(0.2)` $\rightarrow$ `GRU(64)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Output` | Alternativa recorrente computacionalmente mais eficiente que LSTM. |
| **Bi-LSTM** | `Embedding` $\rightarrow$ `SpatialDropout1D(0.2)` $\rightarrow$ `Bidirectional(LSTM(64))` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Output` | Modelação bidirecional do contexto (passado e futuro do diálogo). |
| **Bi-GRU** | `Embedding` $\rightarrow$ `SpatialDropout1D(0.2)` $\rightarrow$ `Bidirectional(GRU(64))` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Output` | Processamento bidirecional com menor complexidade de parâmetros. |
| **CNN 1D** | `Embedding` $\rightarrow$ `SpatialDropout1D(0.2)` $\rightarrow$ `Conv1D(128, kernel=5)` $\rightarrow$ `GlobalMaxPooling1D` $\rightarrow$ `Dense(64)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Output` | Deteção de n-gramas e padrões linguísticos locais de fraude. |

---

## 📁 Estrutura do Repositório

```text
detecao_fraude_voz/
│
├── data/                               # Datasets do projeto
│   ├── MASC (Multi Agent Scam Conversation)/
│   ├── SASC (Single Agent Scam Conversation)/
│   ├── SSC (Scammer Conversation)/
│   ├── SSD (Scam Dialogue)/
│   ├── merged/                         # Datasets combinados (merged, merged_shuffled, merged_half)
│   ├── TeleAntiFraud_Call_Transcript/  # Transcrições reais de chamadas telefónicas
│   └── SOURCES.md                      # Origens e licenças dos dados (Apache 2.0)
│
├── docs/                               # Relatório técnico de licenciatura (PDF)
│   └── RelatorioProjeto_DinisGoncalves.pdf
│
├── experiments/                        # Suítes de experiências isoladas
│   ├── 01_datasets_individuais/
│   │   ├── results/                    # CSVs e gráficos de métricas individuais
│   │   └── script/treinar_datasets_individuais.py
│   ├── 02_dataset_merged/
│   │   ├── results/                    # EDA txt, CSVs e plots comparativos
│   │   └── script/treinar_dataset_merged.py
│   ├── 03_teleantifraud/
│   │   ├── results/                    # Inferências e métricas no dataset real
│   │   └── script/treinar_dataset_teleantifraud.py
│   └── 04_experiencias_extra/
│       ├── results/                    # Heatmaps cross-dataset, early detection e split decay
│       └── script/experiencias_extra.py
│
├── models/                             # Módulos Python das 6 arquiteturas Keras
│   ├── base.py
│   ├── lstm.py
│   ├── gru.py
│   ├── bi_lstm.py
│   ├── bi_gru.py
│   └── cnn.py
│
├── saved_models/                       # Artefactos exportados 
│   └── teleantifraud/                  # cnn_model.keras, tokenizer.json, config.json
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔬 Organização das Experiências & Resultados

### 01. Datasets Sintéticos Individuais (`01_datasets_individuais`)
Treino e avaliação isolada nos 4 datasets sintéticos (*MASC, SASC, SSC, SSD*).
- **Conclusão:** Resultados quase perfeitos ($F1 \approx 1.0$) na maioria dos datasets sintéticos devido à elevada regularidade estrutural dos diálogos gerados. No *MASC* (sequências mais longas), modelos recorrentes unidirecionais falharam ($Recall \approx 0.05$), enquanto **Bi-LSTM**, **Bi-GRU** e **CNN** mantiveram desempenho perfeito.

### 02. Datasets Merged & Sanity Check (`02_dataset_merged`)
Avaliação nos datasets consolidados e desduplicados (*merged*, *merged_shuffled*, *merged_half*). Inclui um mecanismo de controlo:
- **Flag `--sanity-check`:** Baralha aleatoriamente as *labels* no treino (`rng.permutation(y_train)`) mantendo o conjunto de teste intato. Garante que as elevadas métricas obtidas decorrem da aprendizagem real de padrões linguísticos e não de *overfitting* ou memorização de ruído.

### 03. Dataset Real TeleAntiFraud (`03_teleantifraud`)
Avaliação em 13 543 transcrições reais de chamadas telefónicas.
- **Métricas Principais:**
  - **Bi-LSTM:** Maior precisão global ($Accuracy = 0.9989$, $F1 = 0.9990$).
  - **CNN:** Desempenho equivalente ($F1 = 0.9986$) com tempo de treino drasticamente menor.
  - **Modelo Base:** $Precision = 1.0000$ (zero falsos positivos no conjunto de teste).
- **Inferência de Não-Rotulados (`--inferir-unlabeled`):** O modelo **CNN** treinado é exportado para `saved_models/teleantifraud/` (juntamente com o `tokenizer.json` e `config.json` para garantir pré-processamento idêntico) e utilizado para classificar 4 543 instâncias *unlabeled*.

### 04. Experiências Extra & Estudo Avançado (`04_experiencias_extra`)
Script unificado com três baterias de testes focados em robustez operacional:
1. **`todos-contra-todos` (Cross-Dataset Generalization):** Treina num dataset e testa em todos os outros (com *holdout* de 20% no próprio dataset). Mostra que modelos treinados no *TeleAntiFraud* generalizam bem para dados sintéticos, mas o inverso sofre quebras acentuadas de performance.
2. **`menos-palavras` (Early Detection em Tempo Real):** Simula a classificação em *streaming* de chamadas em curso. Trunca as conversas em percentagens (5%, 10%, 20%, 30%, 40%, 50%, 75%, 100% das palavras). Revela que a **CNN** e a **Bi-LSTM** conseguem detetar intenção fraudulenta com $F1 > 0.90$ com apenas 20% a 30% da conversa inicial.
3. **`tamanhos-teste` (Curva de Colapso por Escassez de Dados):** Varia a proporção de teste de 20% a 90% (reduzindo os dados de treino de 80% para 10%) para medir a degradação dos modelos sob restrição severa de dados.

---

## 💻 Guia de Execução (CLI)

### 1. Instalação do Ambiente

```bash
# Clonar o repositório
git clone https://github.com/Dinis-Goncalves/detecao-fraude-voz-dl.git
cd detecao-fraude-voz-dl

# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Comandos Práticos por Experiência

#### 🟢 Experiência 01: Datasets Individuais
```bash
python experiments/01_datasets_individuais/treinar_datasets_individuais.py
```

#### 🟡 Experiência 02: Datasets Merged & Sanity Check
```bash
# Treino padrão (15 épocas com EarlyStopping)
python experiments/02_dataset_merged/treinar_dataset_merged.py --epochs 15

# Teste de Sanity Check (labels baralhadas no treino)
python experiments/02_dataset_merged/treinar_dataset_merged.py --sanity-check --epochs 5
```

#### 🔵 Experiência 03: TeleAntiFraud & Inferência
```bash
# Análise Exploratória (EDA)
python experiments/03_teleantifraud/treinar_dataset_teleantifraud.py --eda

# Treinar os 6 modelos e exportar o modelo vencedor (CNN) para saved_models/
python experiments/03_teleantifraud/treinar_dataset_teleantifraud.py --guardar-modelo --epochs 15

# Classificar instâncias sem rótulo (unlabeled) usando o modelo guardado
python experiments/03_teleantifraud/treinar_dataset_teleantifraud.py --inferir-unlabeled
```

#### 🔴 Experiência 04: Experiências Extra (Generalização / Early Detection / Splits)
```bash
# 1. Matrizes Cross-Dataset (Heatmaps de transferência)
python experiments/04_experiencias_extra/experiencias_extra.py todos-contra-todos --epochs 10

# 2. Deteção Precoce (Chamadas truncadas de 5% a 100%)
python experiments/04_experiencias_extra/experiencias_extra.py menos-palavras --epochs 10

# 3. Curva de Colapso (Variação do split de teste entre 20% e 90%)
python experiments/04_experiencias_extra/experiencias_extra.py tamanhos-teste --epochs 10
```

---

## 🛠️ Reprodutibilidade & Configuração Global

Para garantir resultados reprodutíveis em todas as execuções, é fixada uma *seed* global (`SEED = 42`) no NumPy, Python `random` e TensorFlow:

```python
SEED = 42  # Answer to the Ultimate Question of Life, the Universe, and Everything ;)
```

---