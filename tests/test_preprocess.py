import pandas as pd

from src.data.preprocess import clean_dataframe, preprocess_text


def test_preprocess_text_lowercases():
    assert preprocess_text("HELLO WORLD") == "hello world"


def test_preprocess_text_removes_urls():
    assert "http" not in preprocess_text("visit http://example.com today")


def test_preprocess_text_removes_html():
    assert "<b>" not in preprocess_text("<b>bold</b>")


def test_preprocess_text_strips_mentions():
    assert "@user" not in preprocess_text("Hello @user how are you")


def test_preprocess_text_strips_hashtags():
    assert "#topic" not in preprocess_text("Check out #topic today")


def test_clean_dataframe_removes_duplicates():
    df = pd.DataFrame({
        "text": ["I feel sad", "I feel sad", "I feel great"],
        "status": ["Depression", "Depression", "Normal"],
    })
    result = clean_dataframe(df)
    assert result.duplicated(subset="text").sum() == 0


def test_clean_dataframe_filters_short_texts():
    df = pd.DataFrame({
        "text": ["hi", "I feel very sad today", "ok"],
        "status": ["Normal", "Depression", "Normal"],
    })
    result = clean_dataframe(df, min_word_count=3)
    assert all(len(t.split()) >= 3 for t in result["clean_text"])


def test_clean_dataframe_adds_clean_text_column():
    df = pd.DataFrame({
        "text": ["I feel very sad and hopeless today"],
        "status": ["Depression"],
    })
    result = clean_dataframe(df)
    assert "clean_text" in result.columns
