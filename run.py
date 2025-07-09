import subprocess
import sys
import os
import json
from multiprocessing import Process, Queue
import time

STATUS_FILE_PATH = 'data/scraping_status.json'
LOG_FILE_PATH = 'data/scraping_log.txt'

def update_status(status: str, current_step: str = None, progress: int = None, total_pages: int = None, processed_pages: int = None, error: str = None):
    """スクレイピングの進捗状況をJSONファイルに更新します。"""
    try:
        with open(STATUS_FILE_PATH, 'r+') as f:
            data = json.load(f)
            data["status"] = status
            if current_step is not None: data["current_step"] = current_step
            if progress is not None: data["progress"] = progress
            if total_pages is not None: data["total_pages"] = total_pages
            if processed_pages is not None: data["processed_pages"] = processed_pages
            if error is not None: data["error"] = error
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    except Exception as e:
        print(f"Error updating status file: {e}")

def _run_script_process(cmd_args: list, output_queue: Queue):
    """スクリプトをサブプロセスとして実行し、出力をキューに書き込みます。"""
    process = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        bufsize=1 # 行バッファリング
    )

    # 標準出力をリアルタイムでキューに書き込む
    for line in iter(process.stdout.readline, ''):
        output_queue.put(line)
    process.stdout.close()

    # 標準エラー出力も同様にキューに書き込む
    for line in iter(process.stderr.readline, ''):
        output_queue.put(f"[ERROR] {line}")
    process.stderr.close()

    process.wait()
    output_queue.put(f"--- Script {script_path} finished with exit code {process.returncode} ---")


def main(script_to_run: str = None):
    """プロジェクトのワークフローを実行します。"""
    print(f"Starting the Uma Prediction Project workflow for: {script_to_run if script_to_run else 'Full Workflow'}...")

    # ログファイルをクリア
    with open(LOG_FILE_PATH, 'w') as f:
        f.write("")

    output_queue = Queue() # プロセス間で共有するキュー

    if script_to_run == "outline":
        update_status(status="running", current_step="Starting Scraping Race Overview Data...")
        p = Process(target=_run_script_process, args=([sys.executable, "-m", "scripts.data_acquisition.scraping_outline"], output_queue))
        p.start()
        p.join() # 完了を待つ
    elif script_to_run == "main_scraper":
        update_status(status="running", current_step="Starting Scraping Race Detail Data...")
        p = Process(target=_run_script_process, args=([sys.executable, "-m", "scripts.data_acquisition.main_scraper"], output_queue))
        p.start()
        p.join() # 完了を待つ
    elif script_to_run == "speed_index":
        update_status(status="running", current_step="Starting Speed Index Scraping...")
        p = Process(target=_run_script_process, args=([sys.executable, "-m", "scripts.data_acquisition.scraping_speed_index"], output_queue))
        p.start()
        p.join() # 完了を待つ
    elif script_to_run == "full_workflow":
        update_status(status="running", current_step="Starting Full Scraping Workflow...")
        
        # outlineの実行
        p_outline = Process(target=_run_script_process, args=([sys.executable, "-m", "scripts.data_acquisition.scraping_outline"], output_queue))
        p_outline.start()
        p_outline.join() # outlineの完了を待つ

        if p_outline.exitcode == 0: # outlineが成功した場合のみmain_scraperを実行
            # main_scraperの実行
            p_main = Process(target=_run_script_process, args=([sys.executable, "-m", "scripts.data_acquisition.main_scraper"], output_queue))
            p_main.start()
            p_main.join() # main_scraperの完了を待つ
        else:
            update_status(status="error", error="Scraping outline failed, skipping main scraper.")

    else:
        print("No specific script to run or invalid argument provided.")
        update_status(status="idle", current_step="No specific script to run.")

    # キューに残っている出力をログファイルに書き出す
    while not output_queue.empty():
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(output_queue.get())

    print("\nUma Prediction Project workflow initiated. Check Streamlit UI for status.")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("full_workflow") # デフォルトはフルワークフロー