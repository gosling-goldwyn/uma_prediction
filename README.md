# uma_prediction

## 概要

このプロジェクトは、競馬のレース結果を予測するためのデータ収集、前処理、機械学習モデルのトレーニング、および予測を行うシステムです。Streamlitを用いたWebインターフェースを通じて、スクレイピングの進捗状況を監視・制御できます。

## プロジェクト構造

```:
.
├── app.py                  # Streamlitアプリケーションのメインファイル
├── run.py                  # スクレイピングワークフローの実行を管理
├── pyproject.toml          # プロジェクトのメタデータと依存関係
├── requirements.txt        # Pythonの依存関係リスト
├── data/                   # スクレイピングされたデータ、ステータス、ログ
│   ├── scraping_log.txt
│   ├── scraping_status.json
│   ├── races/              # スクレイピングされたレースデータ（CSV/Parquet）
│   └── dataset.parquet     # 結合されたレースデータ
├── models/                 # 学習済みモデルの保存先
│   ├── rf_uma_prediction_model.pkl # RandomForestモデル
│   └── cnn_uma_prediction_model.h5 # CNNモデル
└── scripts/
    ├── analysis/
    │   └── analyze.py      # データ分析スクリプト
    ├── data_acquisition/   # データスクレイピング関連スクリプト
    │   ├── main_scraper.py
    │   ├── scraping_outline.py
    │   ├── scraping_speed_index.py
    │   └── scraping_util.py
    ├── data_preprocessing/ # データ前処理関連スクリプト
    │   └── adjust_csv.py
    └── model_training/     # モデルトレーニング関連スクリプト
        └── train.py
```

## 機能

* **データスクレイピング:** Netkeiba.comから競馬のレースデータ（概要、詳細）を自動的に収集します。
  * `scraping_outline.py`: レース概要データをスクレイピングします。
  * `main_scraper.py`: レース詳細データをスクレイピングします。途中からの再開機能も備えています。
  * `scraping_speed_index.py`: スピード指数関連のデータをスクレイピングします。
* **データ前処理:** スクレイピングしたデータを機械学習モデルに適した形式に整形・クリーニングします。
* **機械学習モデル:**
  * **RandomForestClassifier:** ベースラインモデルとして、レース結果（1位、2-3位、その他）の多値分類を行います。
  * **Convolutional Neural Network (CNN):** より高度な予測精度を目指すための深層学習モデル。時系列データや構造化データからのパターン認識に利用されます。
* **Streamlit UI:** スクレイピングの進捗状況をリアルタイムで表示し、スクレイピングプロセスの開始やリセットを制御できるユーザーインターフェースを提供します。

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

### 3. TensorFlowのインストール (CNNモデルを使用する場合)

CNNモデルを学習・実行するには、TensorFlowが必要です。

```bash
pip install tensorflow
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

### 3. モデルの学習と推論

スクレイピングが完了し、`data/races`ディレクトリにデータが格納されたら、モデルの学習と推論を行うことができます。

#### モデルの学習

`train.py`を実行してモデルを学習させます。これにより、RandomForestClassifier、LightGBM、およびCNNモデルがトレーニングされ、`models/`ディレクトリに保存されます。CNNモデルのトレーニング時には、推論時に使用する補完値も生成されます。

```bash
uv run -m scripts.model_training.train
```

#### モデルによる推論

学習済みモデルを使用して、Netkeibaの出馬表URLからレース結果を予測します。

```bash
uv run -m scripts.predict_latest_race <Netkeibaの出馬表URL>
```

例:

```bash
uv run -m scripts.predict_latest_race https://race.netkeiba.com/race/shutuba.html?race_id=202405040811
```

`target_mode`を指定することも可能です（`default`または`top3`）。デフォルトは`default`です。

```bash
uv run -m scripts.predict_latest_race <Netkeibaの出馬表URL> top3
```

## 開発者向け情報

### スクリプトの直接実行

`scripts`ディレクトリ内のPythonスクリプトを直接実行する場合、相対インポートの問題を避けるため、モジュールとして実行することを推奨します。

例:

```bash
uv run -m scripts.data_acquisition.main_scraper
uv run -m scripts.data_acquisition.scraping_speed_index
uv run -m scripts.model_training.train
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
