import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from time import sleep
import random  # sleep時間のランダム化のため
import json

STATUS_FILE_PATH = "data/scraping_status.json"


def update_status(
    status: str,
    current_step: str = None,
    progress: int = None,
    total_pages: int = None,
    processed_pages: int = None,
    error: str = None,
):
    """スクレイピングの進捗状況をJSONファイルに更新します。

    Args:
        status (str): 現在のスクレイピングの状態（例: "idle", "running", "completed", "error"）。
        current_step (str, optional): 現在のステップの説明。
        progress (int, optional): 全体に対する進捗率（0-100）。
        total_pages (int, optional): 総ページ数。
        processed_pages (int, optional): 処理済みページ数。
        error (str, optional): エラーメッセージ。
    """
    try:
        with open(STATUS_FILE_PATH, "r+") as f:
            data = json.load(f)
            data["status"] = status
            if current_step is not None:
                data["current_step"] = current_step
            if progress is not None:
                data["progress"] = progress
            if total_pages is not None:
                data["total_pages"] = total_pages
            if processed_pages is not None:
                data["processed_pages"] = processed_pages
            if error is not None:
                data["error"] = error
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    except Exception as e:
        print(f"Error updating status file: {e}")


# 定数
BASE_URL = "https://db.netkeiba.com/"
OUTPUT_PICKLE_PATH = "data/pandas_obj.pkl"

# 元のスクリプトから取得したハードコードされたシリアルとペイロード
# 注意: このserialはセッション依存または時間経過で無効になる可能性があります。
# その場合、このスクリプトは動作しなくなります。
SERIAL = r"a%3A16%3A%7Bs%3A3%3A%22pid%22%3Bs%3A9%3A%22race_list%22%3Bs%3A4%3A%22word%22%3Bs%3A0%3A%22%22%3Bs%3A5%3A%22track%22%3Ba%3A2%3A%7Bi%3A0%3Bs%3A1%3A%221%22%3Bi%3A1%3Bs%3A1%3A%222%22%3B%7Ds%3A10%3A%22start_year%22%3Bs%3A4%3A%222010%22%3Bs%3A9%3A%22start_mon%22%3Bs%3A1%3A%221%22%3Bs%3A8%3A%22end_year%22%3Bs%3A4%3A%22none%22%3Bs%3A7%3A%22end_mon%22%3Bs%3A4%3A%22none%22%3Bs%3A3%3A%22jyo%22%3Ba%3A10%3A%7Bi%3A0%3Bs%3A2%3A%2201%22%3Bi%3A1%3Bs%3A2%3A%2202%22%3Bi%3A2%3Bs%3A2%3A%2203%22%3Bi%3A3%3Bs%3A2%3A%2204%22%3Bi%3A4%3Bs%3A2%3A%2205%22%3Bi%3A5%3Bs%3A2%3A%2206%22%3Bi%3A6%3Bs%3A2%3A%2207%22%3Bi%3A7%3Bs%3A2%3A%2208%22%3Bi%3A8%3Bs%3A2%3A%2209%22%3Bi%3A9%3Bs%3A2%3A%2210%22%3B%7Ds%3A9%3A%22kyori_min%22%3Bs%3A0%3A%22%22%3Bs%3A9%3A%22kyori_max%22%3Bs%3A0%3A%22%22%3Bs%3A4%3A%22sort%22%3Bs%3A4%3A%22date%22%3Bs%3A4%3A%22list%22%3Bs%3A3%3A%22100%22%3Bs%3A9%3A%22style_dir%22%3Bs%3A17%3A%22style%2Fnetkeiba.ja%22%3Bs%3A13%3A%22template_file%22%3Bs%3A14%3A%22race_list.html%22%3Bs%3A9%3A%22style_url%22%3Bs%3A18%3A%22%2Fstyle%2Fnetkeiba.ja%22%3Bs%3A6%3A%22search%22%3Bs%3A113%3A%22%B6%A5%C1%F6%BC%EF%CA%CC%5B%BC%C7%A1%A2%A5%C0%A1%BC%A5%C8%5D%A1%A2%B4%FC%B4%D6%5B2010%C7%AF1%B7%EE%A1%C1%CC%B5%BB%D8%C4%EA%5D%A1%A2%B6%A5%C7%CF%BE%EC%5B%BB%A5%CB%DA%A1%A2%C8%A1%B4%DB%A1%A2%CA%A1%C5%E7%A1%A2%BF%B7%B3%E3%A1%A2%C5%EC%B5%FE%A1%A2%C3%E6%BB%B3%A1%A2%C3%E6%B5%FE%A1%A2%B5%FE%C5%D4%A1%A2%BA%E5%BF%C0%A1%A2%BE%AE%C1%D2%5D%22%3B%7D"
INITIAL_PAYLOAD = r"pid=race_list&word&track[]=1&track[]=2&start_year=2010&start_mon=1&end_year=none&end_mon=none&jyo[]=01&jyo[]=02&jyo[]=03&jyo[]=04&jyo[]=05&jyo[]=06&jyo[]=07&jyo[]=08&jyo[]=09&jyo[]=10&kyori_min&kyori_max&sort=date&list=100"
QUERY = {
    "pid": "race_list",
    "word": "",
    "track[]": ["1", "2"],
    "start_year": "2010",
    "start_mon": "1",
    "end_year": "none",
    "end_mon": "none",
    "jyo[]": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
    "kyori_min": "",
    "kyori_max": "",
    "sort": "date",
    "list": "100",
    "track": ["1", "2"],
    "jyo": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,like Gecko) Chrome/XX.0.0.0 Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded",
    "Cookie": "url=http%3A%2F%2Fdb.netkeiba.com%2F",
}


def fetch_page_post(
    url: str, headers: dict, data: str, min_sleep: float = 5.0, max_sleep: float = 10.0
) -> BeautifulSoup:
    """指定されたURLにPOSTリクエストを送信し、BeautifulSoupオブジェクトを返します。

    サーバーへの負荷を考慮し、リクエスト後にランダムな時間スリープします。

    Args:
        url (str): 取得するページのURL。
        headers (dict): リクエストヘッダー。
        data (str): POSTデータ。
        min_sleep (float): スリープ時間の最小値（秒）。
        max_sleep (float): スリープ時間の最大値（秒）。

    Returns:
        BeautifulSoup: ページのBeautifulSoupオブジェクト。取得に失敗した場合はNone。
    """
    try:
        res = requests.post(url, headers=headers, data=data)
        res.raise_for_status()  # HTTPエラーがあれば例外を発生させる
        res.encoding = "shift-jis"  # netkeiba.comはShift-JISの可能性
        sleep(random.uniform(min_sleep, max_sleep))
        return BeautifulSoup(res.content, features="html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url} with POST data {data}: {e}")
        return None


def parse_race_overview_table(soup: BeautifulSoup) -> list:
    """レース概要ページのBeautifulSoupオブジェクトからレース一覧テーブルをパースします。

    Args:
        soup (BeautifulSoup): レース概要ページのBeautifulSoupオブジェクト。

    Returns:
        list: 各レースのデータ（リストのリスト）。ヘッダー行は含まない。
    """
    race_table = soup.find("table", class_="race_table_01")
    if not race_table:
        return []

    rows_data = []
    # ヘッダー行を取得
    headers = [th.text.strip() for th in race_table.find_all("th")]
    if headers:  # ヘッダーが存在すれば最初の要素として追加
        rows_data.append(headers)

    # データ行を取得
    for tr in race_table.find_all("tr")[1:]:  # ヘッダー行はスキップ
        row = []
        for td in tr.find_all("td"):
            # レース名とURLはaタグから取得
            race_link = td.find("a", href=re.compile(r"/race/[0-9]+/"))
            if race_link:
                row.append(race_link.get("href"))
            else:
                row.append(td.text.strip().replace("\n", ""))
        if row:  # 空の行を除外
            rows_data.append(row)

    return rows_data


def get_total_pages(soup: BeautifulSoup) -> int:
    """ページネーションから総ページ数を取得します。

    Args:
        soup (BeautifulSoup): ページのBeautifulSoupオブジェクト。

    Returns:
        int: 総ページ数。見つからない場合は1を返します。
    """
    pager = soup.find("div", class_="common_pager")  # ページネーションのdivを取得
    pager = pager.find_all("ul")[-1]  # 最後のulを取得（最終ページのリンクが含まれるul）
    if pager:
        for e in pager.find_all("li"):
            if e.text.find("最後") >= 0:
                print(e.find("a"))
                last_page_link = e.find("a").attrs["href"]
                break
        else:
            return 1  # "最後"が見つからない場合は1ページとみなす
        idx = last_page_link.find("page=")
        return int(last_page_link[idx + len("page=") :])
    return 1  # ページネーションが見つからない場合は1ページとみなす


def main():
    """レース概要スクレイピングのメイン関数。"""
    update_status(
        status="running", current_step="Scraping Race Overview Data", progress=0
    )
    all_merged_data = []
    current_page = 1
    total_pages = 1  # 初期値

    try:
        while current_page <= total_pages:
            print(f"Scraping page {current_page}/{total_pages}...")
            update_status(
                status="running",
                current_step=f"Scraping page {current_page}/{total_pages}",
                processed_pages=current_page,
                total_pages=total_pages,
            )

            if current_page == 1:
                payload = INITIAL_PAYLOAD
            else:
                payload = f"sort_key=date&sort_type=desc&page={current_page}&serial={SERIAL}&pid=race_list"

            soup = fetch_page_post(BASE_URL, HEADERS, payload)

            if soup is None:
                print(f"Failed to fetch page {current_page}. Exiting.")
                update_status(
                    status="error", error=f"Failed to fetch page {current_page}."
                )
                break

            page_data = parse_race_overview_table(soup)
            if not page_data or (
                len(page_data) == 1 and page_data[0][0] == "開催日"
            ):  # ヘッダーのみの場合も終了
                print(f"No more race data found on page {current_page}. Exiting.")
                update_status(
                    status="completed",
                    current_step=f"No more race data found on page {current_page}. Exiting.",
                )
                break

            # 最初のページで総ページ数を取得
            if current_page == 1:
                total_pages = get_total_pages(soup)
                print(f"Total pages found: {total_pages}")
                update_status(status="running", total_pages=total_pages)

            # ヘッダー行は最初のページからのみ取得し、それ以降はデータのみ追加
            if current_page == 1:
                all_merged_data.extend(page_data)
            else:
                all_merged_data.extend(page_data[1:])  # ヘッダー行を除いて追加

            current_page += 1
            update_status(
                status="running", progress=int(current_page / total_pages * 100)
            )

        if not all_merged_data:
            print("No data scraped.")
            update_status(status="completed", current_step="No data scraped.")
            return

        # DataFrameの作成
        # 最初の行がヘッダーであることを前提とする
        if len(all_merged_data) > 1:
            df = pd.DataFrame(
                data=np.array(all_merged_data[1:]), columns=all_merged_data[0]
            )
            df.to_pickle(OUTPUT_PICKLE_PATH)
            print(f"All overview data saved to {OUTPUT_PICKLE_PATH}")
            update_status(
                status="completed",
                current_step="Race overview data scraped and saved.",
                progress=100,
            )
        else:
            print("Only header found, no data to save.")
            update_status(
                status="completed", current_step="Only header found, no data to save."
            )

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        update_status(status="error", error=str(e))


if __name__ == "__main__":
    main()
