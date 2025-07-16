from playwright.sync_api import sync_playwright
import pandas as pd
from bs4 import BeautifulSoup
import time

def fetch_race_data_with_playwright(race_url: str):
    """
    Playwrightを使用してレースページのHTMLを取得し、出馬表とオッズ情報を解析します。
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            # ページに移動し、DOMの準備が完了するまで待つ
            page.goto(race_url, wait_until='domcontentloaded', timeout=60000)

            # ポップアップやモーダルを閉じるためにEscapeキーを押す
            page.keyboard.press('Escape')
            time.sleep(1) # 念のため少し待つ

            # --- 出馬表とオッズを同時に取得 ---
            try:
                # 出馬表のテーブル内で、オッズが表示されるのを待つ
                page.wait_for_selector('table.Shutuba_Table td.Popular', timeout=20000)
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')

                horse_info_table = soup.find('table', class_='Shutuba_Table')
                if not horse_info_table:
                    raise ValueError("Could not find Shutuba_Table after waiting.")

                rows = horse_info_table.find_all('tr')[1:]
                horse_data = []
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) < 10: # オッズのセルまであるか確認
                        continue

                    try:
                        # オッズと人気を取得
                        odds_cell = cells[9] # 10番目のセルがオッズ
                        odds = float(odds_cell.find('span').text)
                        
                        horse_num = int(cells[1].text.strip())
                        horse_data.append({
                            'horse_num': horse_num,
                            'horse_name': cells[3].find('a').text.strip(),
                            'sex_age': cells[4].text.strip(),
                            'weight_carry': cells[5].text.strip(),
                            'jockey': cells[6].find('a').text.strip(),
                            'horse_weight_change': cells[8].text.strip(), # 9番目のセルが馬体重
                            'odds': odds
                        })
                    except (ValueError, IndexError, AttributeError) as e:
                        print(f"Skipping a row due to parsing error: {e}, row: {row}")
                        continue
                
                df_detail = pd.DataFrame(horse_data)

            except Exception as e:
                print(f"Could not find or parse Shutuba_Table with odds: {e}")
                # エラーが発生しても、基本的な出馬表の解析を試みる
                print("Falling back to basic Shutuba_Table parsing without odds.")
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')
                df_detail, soup = parse_basic_shutuba_table(soup)

            browser.close()
            return df_detail, soup

        except Exception as e:
            print(f"An error occurred during Playwright scraping: {e}")
            browser.close()
            return pd.DataFrame(), None

def parse_basic_shutuba_table(soup: BeautifulSoup):
    """
    オッズ情報なしで、基本的な出馬表テーブルのみを解析するフォールバック関数。
    """
    try:
        horse_info_table = soup.find('table', class_='Shutuba_Table')
        if not horse_info_table:
            print("Fallback failed: Could not find Shutuba_Table.")
            return pd.DataFrame(), soup

        rows = horse_info_table.find_all('tr')[1:]
        horse_data = []
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 8:
                continue
            
            try:
                horse_data.append({
                    'horse_num': int(cells[1].text.strip()),
                    'horse_name': cells[3].find('a').text.strip(),
                    'sex_age': cells[4].text.strip(),
                    'weight_carry': cells[5].text.strip(),
                    'jockey': cells[6].find('a').text.strip(),
                    'horse_weight_change': cells[8].text.strip(), # 9番目のセル
                    'odds': 0.0  # オッズは0で埋める
                })
            except (ValueError, IndexError, AttributeError) as e:
                print(f"Skipping a row in fallback parser due to error: {e}")
                continue
        
        return pd.DataFrame(horse_data), soup
    except Exception as e:
        print(f"An error occurred in fallback parser: {e}")
        return pd.DataFrame(), soup
