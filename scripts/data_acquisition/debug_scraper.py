import sys
from scripts.prediction_utils.scraper import fetch_race_data_with_playwright

def debug_scraping(race_url: str):
    print(f"--- Starting Playwright Scraper Debug for URL: {race_url} ---")
    df_detail, soup = fetch_race_data_with_playwright(race_url)

    if df_detail.empty or not soup:
        print("\n--- Debug Finished: Failed to retrieve data. ---")
        return

    print("\n--- Successfully fetched data. ---")
    print("\n[DataFrame Head]")
    print(df_detail.head())

    print("\n[DataFrame Info]")
    df_detail.info()

    print("\n[Odds Data Check]")
    print(df_detail[['horse_name', 'odds']])

    # Check for missing odds
    if df_detail['odds'].isnull().any() or (df_detail['odds'] == 0).any():
        print("\nWarning: Some odds are missing or zero.")
    else:
        print("\nSuccess: All odds seem to be captured.")

    print("\n--- Debug Finished ---")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
        debug_scraping(url)
    else:
        print("Usage: python -m scripts.data_acquisition.debug_scraper <race_url>")
        print("Example: python -m scripts.data_acquisition.debug_scraper https://race.netkeiba.com/race/shutuba.html?race_id=202405040811")
