import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL_PREFIX = "https://db.netkeiba.com"

def fetch_page(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36)"
    }
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        res.encoding = "EUC-JP"
        return BeautifulSoup(res.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_race_details(soup: BeautifulSoup) -> pd.DataFrame:
    table = soup.find("table", class_="Shutuba_Table")
    if not table:
        return pd.DataFrame()

    all_rows = table.find_all("tr")
    
    header_rows = all_rows[:2]
    data_rows = all_rows[2:]

    headers_level1 = [th.get_text(strip=True) for th in header_rows[0].find_all("th")]
    
    final_headers = [
        "枠", "馬番", "印", "馬名", "性齢", "斤量", "騎手", "厩舎", "馬体重(増減)", "オッズ", "人気",
        "登録", "グループ", "馬メモ", "マスターレース別馬メモ"
    ]

    data = []
    for row in data_rows:
        cols = row.find_all("td")
        if not cols:
            continue
        
        row_data = []
        for i, col in enumerate(cols):
            odds_span = col.find("span", id=re.compile(r"odds-1_\d+"))
            if odds_span:
                row_data.append(odds_span.get_text(strip=True))
            else:
                row_data.append(col.get_text(strip=True))
        data.append(row_data)

    df = pd.DataFrame(data, columns=final_headers)

    df.rename(
        columns={
            "枠": "waku",
            "馬番": "horse_num",
            "印": "check_mark",
            "馬名": "horse_name",
            "性齢": "sex_age",
            "斤量": "weight_carry",
            "騎手": "jockey",
            "厩舎": "stable",
            "馬体重(増減)": "horse_weight_change",
            "人気": "popularity",
            "オッズ": "odds",
            "登録": "fav_regist",
            "グループ": "fav_group",
            "馬メモ": "horse_memo",
            "マスターレース別馬メモ": "master_horse_memo",
        },
        inplace=True,
    )

    horse_num_col_candidates = [
        col for col in df.columns if "horse_num" in col
    ]
    if not horse_num_col_candidates:
        print("Error: 'horse_num' column not found in parsed table.")
        return pd.DataFrame()
    horse_num_col = horse_num_col_candidates[0]

    df = df[df[horse_num_col].astype(str).str.isdigit()]

    sex_age_col_candidates = [col for col in df.columns if "sex_age" in col]
    if not sex_age_col_candidates:
        print("Error: 'sex_age' column not found in parsed table.")
        return pd.DataFrame()
    sex_age_col = sex_age_col_candidates[0]

    return df

if __name__ == "__main__":
    test_url = "https://race.netkeiba.com/race/shutuba.html?race_id=202405040811"
    print(f"Fetching and parsing: {test_url}")
    soup = fetch_page(test_url)
    if soup:
        df_details = parse_race_details(soup)
        if not df_details.empty:
            print("\n--- Parsed Race Details (df_details) ---")
            print(df_details[['horse_name', 'odds', 'popularity', 'horse_memo', 'master_horse_memo']].head())
            print("DataFrame columns:", df_details.columns.tolist())
            print("DataFrame shape:", df_details.shape)
        else:
            print("DataFrame is empty.")
    else:
        print("Failed to fetch page.")