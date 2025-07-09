import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep


def fetch_page(url: str, sleep_time: int = 5) -> BeautifulSoup:
    """指定されたURLのページを取得し、BeautifulSoupオブジェクトを返します。

    サーバーへの負荷を考慮し、リクエスト後に指定時間スリープします。

    Args:
        url (str): 取得するページのURL。
        sleep_time (int, optional): リクエスト後のスリープ時間（秒）。デフォルトは1。

    Returns:
        BeautifulSoup: ページのBeautifulSoupオブジェクト。取得に失敗した場合はNone。
    """
    try:
        HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,like Gecko) Chrome/XX.0.0.0 Safari/537.36",
        }
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()  # HTTPエラーがあれば例外を発生させる
        res.encoding = 'EUC-JP' # Netkeiba.comはEUC-JPを使用していることが多い
        sleep(sleep_time)
        return BeautifulSoup(res.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        sleep(sleep_time)
        return None


def parse_race_details(soup: BeautifulSoup) -> pd.DataFrame:
    """レース詳細ページのBeautifulSoupオブジェクトからレース結果をパースします。

    Args:
        soup (BeautifulSoup): レース詳細ページのBeautifulSoupオブジェクト。

    Returns:
        pd.DataFrame: レース結果のDataFrame。テーブルが見つからない場合は空のDataFrame。
    """
    # class="race_table_01"を持つテーブルを探す
    race_table = soup.find("table", class_="race_table_01")
    if not race_table:
        return pd.DataFrame()

    # テーブルのヘッダー（thタグ）を取得
    headers = [th.text.strip() for th in race_table.find_all("th")]

    # テーブルの各行（trタグ）からデータを取得
    rows = []
    for tr in race_table.find_all("tr")[1:]:  # ヘッダー行はスキップ
        cells = [td.text.strip().replace("\n", "") for td in tr.find_all("td")]
        if len(cells) == len(headers):
            rows.append(cells)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=headers)
    return df
