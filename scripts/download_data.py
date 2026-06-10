from __future__ import annotations

from jepa_trading.config import ensure_dirs, load_config
from jepa_trading.data.pipeline import prepare_market_data


def main() -> None:
    config = load_config()
    ensure_dirs(config)
    df, arrays, feature_columns = prepare_market_data(config)
    print("prepared rows:", len(df))
    print("assets:", len(arrays.tickers), arrays.tickers[:10])
    print("features:", len(feature_columns), feature_columns)


if __name__ == "__main__":
    main()

