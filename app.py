import streamlit as st
import json
import time
import subprocess
import os
import sys

STATUS_FILE_PATH = 'data/scraping_status.json'
LOG_FILE_PATH = 'data/scraping_log.txt'
TRAINING_STATUS_FILE = 'data/training_status.json'

def get_status():
    """スクレイピングのステータスをファイルから読み込みます。"""
    if not os.path.exists(STATUS_FILE_PATH):
        return {
            "status": "idle",
            "current_step": "No status file found.",
            "progress": 0,
            "total_pages": 0,
            "processed_pages": 0,
            "error": "None"
        }
    with open(STATUS_FILE_PATH, 'r') as f:
        return json.load(f)

def reset_status():
    """ステータスファイルを初期状態にリセットします。"""
    initial_status = {
        "status": "idle",
        "current_step": "None",
        "progress": 0,
        "total_pages": 0,
        "processed_pages": 0,
        "error": "None"
    }
    with open(STATUS_FILE_PATH, 'w') as f:
        json.dump(initial_status, f, indent=4)
    # ログファイルもクリア
    with open(LOG_FILE_PATH, 'w') as f:
        f.write("")

def run_workflow(script_type: str = "full_workflow"):
    """run.pyスクリプトをバックグラウンドで実行します。"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
    
    reset_status()

    # run.py をバックグラウンドで実行
    # stdoutとstderrはrun.py内で処理されるため、ここではPIPEにリダイレクトしない
    subprocess.Popen([sys.executable, script_path, script_type])

def get_training_status():
    """学習のステータスをファイルから読み込みます。"""
    if not os.path.exists(TRAINING_STATUS_FILE):
        return {
            "status": "idle",
            "current_model": "None",
            "progress": 0,
            "message": "No training status file found."
        }
    with open(TRAINING_STATUS_FILE, 'r') as f:
        return json.load(f)

def run_training():
    """train.pyスクリプトをバックグラウンドで実行します。"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "model_training", "train.py")
    
    # 既存のステータスファイルをリセット
    initial_status = {
        "status": "idle",
        "current_model": "None",
        "progress": 0,
        "message": "Starting training..."
    }
    with open(TRAINING_STATUS_FILE, 'w') as f:
        json.dump(initial_status, f, indent=4)

    subprocess.Popen([sys.executable, script_path])

st.set_page_config(page_title="Uma Prediction Scraping Status", layout="wide")
st.title("🐎 Uma Prediction Status")

# サイドバー
st.sidebar.header("Controls")

st.sidebar.markdown("### Scraping Workflow")
if st.sidebar.button("Start Full Scraping Workflow"): 
    run_workflow("full_workflow")
    st.sidebar.success("Full scraping workflow started!")

st.sidebar.markdown("### Run Individual Scraping Steps")
if st.sidebar.button("Run Scraping Outline"): 
    run_workflow("outline")
    st.sidebar.success("Scraping Outline started!")

if st.sidebar.button("Run Main Scraper"): 
    run_workflow("main_scraper")
    st.sidebar.success("Main Scraper started!")

if st.sidebar.button("Run Speed Index Scraping"): 
    run_workflow("speed_index")
    st.sidebar.success("Speed Index Scraping started!")

if st.sidebar.button("Reset Scraping Status"): 
    reset_status()
    st.sidebar.info("Scraping status reset.")

st.sidebar.markdown("### Model Training")
if st.sidebar.button("Start Model Training"): 
    run_training()
    st.sidebar.success("Model training started!")

# メインコンテンツ
scraping_status_placeholder = st.empty()
scraping_progress_bar_placeholder = st.empty()
scraping_log_placeholder = st.empty() # ログ表示用のプレースホルダー

st.markdown("--- ")

training_status_placeholder = st.empty()
training_progress_bar_placeholder = st.empty()
training_report_placeholder = st.empty()

while True:
    # スクレイピングステータスの表示
    status_data = get_status()
    
    with scraping_status_placeholder.container():
        st.subheader("Current Scraping Status")
        st.json(status_data)

    with scraping_progress_bar_placeholder.container():
        if status_data["status"] == "running":
            st.progress(status_data["progress"] / 100.0, text=f"Progress: {status_data["progress"]}%")
        elif status_data["status"] == "completed":
            st.success("Scraping Completed!")
            st.progress(1.0, text="Progress: 100%")
        elif status_data["status"] == "error":
            st.error(f"Scraping Error: {status_data["error"]}")
            st.progress(0.0, text="Progress: 0%")
        else:
            st.info("Scraping: Idle. Click a button to begin.")
            st.progress(0.0, text="Progress: 0%")

    # ログファイルのリアルタイム表示
    with scraping_log_placeholder.container():
        st.subheader("Scraping Log")
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, 'r') as f:
                log_content = f.read()
            st.code(log_content, language='text')
        else:
            st.info("Log file not found yet.")

    # 学習ステータスの表示
    training_status_data = get_training_status()

    with training_status_placeholder.container():
        st.subheader("Current Model Training Status")
        st.json(training_status_data)

    with training_progress_bar_placeholder.container():
        if training_status_data["status"] == "running":
            st.progress(training_status_data["progress"] / 100.0, text=f"Training Progress ({training_status_data["current_model"] or ""}): {training_status_data["progress"]}%")
        elif training_status_data["status"] == "completed":
            st.success("Model Training Completed!")
            st.progress(1.0, text="Training Progress: 100%")
        elif training_status_data["status"] == "error":
            st.error(f"Model Training Error: {training_status_data["message"]}")
            st.progress(0.0, text="Training Progress: 0%")
        else:
            st.info("Model Training: Idle. Click a button to begin.")
            st.progress(0.0, text="Training Progress: 0%")

    with training_report_placeholder.container():
        if "report" in training_status_data:
            st.subheader("Training Report")
            st.json(training_status_data["report"])

    time.sleep(1) # 1秒ごとに更新