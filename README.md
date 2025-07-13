# uma_prediction

## 概要

このプロジェクトは、競馬のレース結果を予測するためのデータ収集、前処理、機械学習モデルのトレーニング、および予測を行うシステムです。Streamlitを用いたWebインターフェースを通じて、スクレイピングの進捗状況を監視・制御できます。

## プロジェクト構造

```
.
├── app.py                  # Streamlitアプリケーションのメインファイル
├── run.py                  # スクレイピングワークフローの実行を管理
├── pyproject.toml          # プロジェクトのメタデータと依存関係
├── requirements.txt        # Pythonの依存関係リスト
├── upload_dataset_to_hf.py # Hugging Face Datasetsへデータセットをアップロードするスクリプト
├── upload_models_to_hf.py  # Hugging Face Hubへ学習済みモデルをアップロードするスクリプト
├── data/                   # スクレイピングされたデータ、ステータス、ログ
│   ├── scraping_log.txt
│   ├── scraping_status.json
│   ├── races/              # スクレイピングされたレースデータ（CSV/Parquet）
│   └── index_dataset.parquet # 結合されたレースデータ
├── models/                 # 学習済みモデルの保存先
└── scripts/
    ├── analysis/
    │   └── analyze.py      # データ分析スクリプト
    ├── data_acquisition/   # データスクレイピング関連スクリプト
    │   ├── debug_scraper.py # スクレイピングデバッグ用
    │   ├── main_scraper.py
    │   ├── scraping_outline.py
    │   ├── scraping_speed_index.py
    │   └── scraping_util.py
    ├── data_preprocessing/ # データ前処理関連スクリプト
    │   ├── adjust_csv.py
    │   └── lgbm_categorical_processor.py # LGBMカテゴリカル特徴量処理用
    ├── model_training/     # モデルトレーニング関連スクリプト
    │   ├── train.py        # ローカル学習用
    │   ├── train_colab.py  # Colab+Hugging Face学習用
    │   └── training_utils.py
    └── prediction_utils/   # 推論関連ユーティリティ
        ├── constants.py
        ├── data_preprocessor.py
        ├── ensembler.py
        ├── model_loader.py
        ├── predictor.py
        └── value_betting.py
```

## 機能

### 1. スクレイピングフェーズ

Netkeiba.comから競馬のレースデータを自動的に収集します。

*   `scripts/data_acquisition/scraping_outline.py`: レース概要データをスクレイピングします。
*   `scripts/data_acquisition/main_scraper.py`: レース詳細データをスクレイピングします。途中からの再開機能も備えています。
*   `scripts/data_acquisition/scraping_speed_index.py`: スピード指数関連のデータをスクレイピングします。
*   `scripts/data_acquisition/debug_scraper.py`: スクレイピングロジックのデバッグに特化したスクリプトです。

### 2. 学習フェーズ

スクレイピングされたデータを用いて機械学習モデルを学習します。

*   **ローカルでの学習**: `scripts/model_training/train.py` を使用して、RandomForest、LightGBM、およびCNNモデルをトレーニングし、`models/`ディレクトリに保存します。
*   **Colab + Hugging Faceでの学習**: `scripts/model_training/train_colab.py` を使用します。このスクリプトは、Hugging Face Datasetsからデータをロードし、学習後にモデルと関連メタデータをHugging Face Hubに自動的にアップロードします。
    *   `upload_dataset_to_hf.py`: ローカルのデータセット (`data/index_dataset.parquet`) をHugging Face Datasetsにアップロードするために使用します。
    *   `upload_models_to_hf.py`: ローカルの学習済みモデルをHugging Face Hubにアップロードするために使用します。

### 3. 推論フェーズ

学習済みモデルを使用して、Netkeibaの出馬表URLからレース結果を予測します。

*   **アンサンブル推論**: `scripts/predict_ensemble.py` を使用し、複数のモデルの予測を統合して最終的な予測結果を出力します。バリューベッティング分析も行います。
*   **各モデル独立推論**: `scripts/predict_latest_race.py` を使用し、各モデル（RandomForest、LightGBM、CNN）が算出した個別の予測確率を出力します。
*   **手動入力推論**: `scripts/predict_manual_input.py` を使用し、ユーザーが手動で入力したデータに基づいて予測を実行します。

### その他

*   **データ前処理**: スクレイピングしたデータを機械学習モデルに適した形式に整形・クリーニングします。
*   **機械学習モデル**: RandomForestClassifier、LightGBM、Convolutional Neural Network (CNN) を使用します。
*   **Streamlit UI**: スクレイピングの進捗状況をリアルタイムで表示し、スクレイピングプロセスの開始やリセットを制御できるユーザーインターフェースを提供します。

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/your_username/uma_prediction.git
cd uma_prediction
```

### 2. Python環境のセットアップ

**重要: TensorFlowは現在 (2025/7/8) Python 3.13に対応していません。Python 3.12を使用してください。**

`uv` を使用して依存関係をインストールすることを推奨します。

```bash
uv sync
```

または `pip` を使用する場合:

```bash
pip install -r requirements.txt
```

### 3. Hugging Face APIトークンの設定

Hugging Face Hubとの連携にはAPIトークンが必要です。`HF_TOKEN` および `REPO_ID`、`DATASET_REPO_ID` を `.env` ファイルに設定してください。

```
HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"
REPO_ID="your_username/uma_prediction_models"
DATASET_REPO_ID="your_username/uma_dataset"
```

## 使い方

### 1. Streamlitアプリケーションの起動

スクレイピングの進捗監視と制御のために、Streamlitアプリケーションを起動します。

```bash
streamlit run app.py
```

ブラウザで表示されるURLにアクセスしてください。

### 2. スクレイピングの実行

Streamlit UIから「Start Full Scraping Workflow」ボタンをクリックするか、個別のスクレイピングステップを実行できます。

スクレイピングは`data/scraping_status.json`と`data/scraping_log.txt`で進捗が管理されます。`main_scraper.py`は、中断しても途中から再開できるロジックが組み込まれています。

*   **スピード指数データのスクレイピング:**
    ```bash
    uv run -m scripts.data_acquisition.scraping_speed_index
    ```

### 3. モデルの学習

#### ローカルでの学習

```bash
uv run -m scripts.model_training.train
```

#### Colab + Hugging Faceでの学習

1.  **データセットのアップロード**: `data/index_dataset.parquet` をHugging Face Datasetsにアップロードします。
    ```bash
    uv run python upload_dataset_to_hf.py
    ```
2.  **Colabで学習スクリプトを実行**: `scripts/model_training/train_colab.py` をColab環境で実行します。このスクリプトは自動的にデータをロードし、学習後にモデルをHugging Face Hubにアップロードします。

### 4. 学習済みモデルのダウンロード

Hugging Face Hubから最新の学習済みモデルをダウンロードします。

```bash
uv run python download_models.py
```

### 5. モデルによる推論

#### アンサンブル推論

```bash
uv run -m scripts.predict_ensemble <Netkeibaの出馬表URL> [target_mode]
```

例:

```bash
uv run -m scripts.predict_ensemble https://race.netkeiba.com/race/shutuba.html?race_id=202405040811
```

`target_mode`を指定することも可能です（`default`または`top3`）。デフォルトは`default`です。

#### 各モデル独立推論

```bash
uv run -m scripts.predict_latest_race <Netkeibaの出馬表URL> [target_mode]
```

例:

```bash
uv run -m scripts.predict_latest_race https://race.netkeiba.com/race/shutuba.html?race_id=202405040811
```

`target_mode`を指定することも可能です（`default`または`top3`）。デフォルトは`default`です。

#### 手動入力推論

```bash
uv run -m scripts.predict_manual_input
```

## 開発者向け情報

### スクリプトの直接実行

`scripts`ディレクトリ内のPythonスクリプトを直接実行する場合、モジュールとして実行することを推奨します。

例:

```bash
uv run -m scripts.data_acquisition.main_scraper
uv run -m scripts.data_acquisition.scraping_speed_index
uv run -m scripts.model_training.train
uv run -m scripts.predict_ensemble https://race.netkeiba.com/race/shutuba.html?race_id=202405040811
uv run -m scripts.predict_latest_race https://race.netkeiba.com/race/shutuba.html?race_id=202405040811
```

## 貢献

このプロジェクトへの貢献を歓迎します。バグ報告、機能提案、プルリクエストなど、お気軽にお寄せください。

## 免責事項

本プロジェクトに含まれるスクレイピング機能は、netkeiba.comのデータを収集するために設計されています。netkeiba.comの利用規約では、サービスに支障をきたすようなスクレイピング行為を禁止しており、これに違反した場合、アクセス制限を受ける可能性があります。

**アクセス制限を受けた場合、その制限は解除されない可能性もあります。**

本プロジェクトの利用者は、netkeiba.comの利用規約を遵守する責任があります。本プロジェクトの利用によって生じたいかなる損害や問題についても、プロジェクトの作成者および貢献者は一切の責任を負いません。自己責任においてご利用ください。

**情報源:** [netkeiba.com ヘルプ - データベースの閲覧ができない・通信制限がかかった（スクレイピングについて）](https://support.keiba.netkeiba.com/hc/ja/articles/18841959592857-%E3%83%87%E3%83%BC%E3%82%BF%E3%83%99%E3%83%BC%E3%82%B9%E3%81%AE%E9%96%B2%E8%A6%A7%E3%81%8C%E3%81%A7%E3%81%8D%E3%81%AA%E3%81%84-%E9%80%9A%E4%BF%A1%E5%88%B6%E9%99%90%E3%81%8C%E3%81%8B%E3%81%8B%E3%81%A3%E3%81%9F-%E3%82%B9%E3%82%AF%E3%83%AC%E3%82%A4%E3%83%93%E3%83%B3%E3%82%B0%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6)

## ライセンス

[ここにライセンス情報を記述]