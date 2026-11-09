from src.hm import load_hm


def test_load_hm_timestamps_are_milliseconds_and_repeats_collapse(tmp_path):
    path = tmp_path / "transactions_train.csv"
    path.write_text(
        "t_dat,customer_id,article_id,price,sales_channel_id\n"
        "2020-01-01,c1,0108775015,0.1,2\n"
        "2020-01-02,c1,0108775015,0.1,2\n"  # repeat purchase, collapsed to the first
        "2020-01-02,c2,0108775044,0.2,1\n"
    )
    df = load_hm(str(path), last_weeks=None)
    assert len(df) == 2
    assert df["item_id"].tolist() == ["108775015", "108775044"]  # leading zero dropped
    assert df["timestamp"].iloc[0] == 1577836800000  # 2020-01-01 00:00 UTC in ms
