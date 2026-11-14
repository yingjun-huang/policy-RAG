"""Unit tests for text normalization and SimHash deduplication."""

from policy_rag.indexing.cleaning import hamming64, normalize_text, simhash64


def test_normalize_text_removes_zero_width_and_control_chars():
    dirty = "退换货\u200b政策\ufeff测试\x00内容"
    cleaned = normalize_text(dirty)
    assert cleaned == "退换货政策测试内容"


def test_normalize_text_joins_cjk_line_breaks():
    text_with_cjk_breaks = "七天无\n理由退换货\n规则"
    cleaned = normalize_text(text_with_cjk_breaks, join_hard_breaks=True)
    assert cleaned == "七天无理由退换货规则"


def test_normalize_nfkc_and_whitespace():
    raw = "７天无理由退货   （满５９包邮）\n\n\n\n偏远地区"
    cleaned = normalize_text(raw)
    assert "7天无理由退货 (满59包邮)" in cleaned
    assert "\n\n" in cleaned  # collapsed to double newline
    assert "\n\n\n" not in cleaned


def test_simhash_near_duplicates():
    text1 = "非定制未拆封影响二次销售的商品签收次日起7个自然日内可申请无理由退货。"
    text2 = "非定制未拆封影响二次销售的商品签收次日起7个自然日内可申请无理由退货！"
    text3 = "西藏新疆青海等偏远地区订单需额外加收15元远程物流运费不包邮。"

    h1 = simhash64(text1)
    h2 = simhash64(text2)
    h3 = simhash64(text3)

    # Near duplicates should have small hamming distance
    dist_1_2 = hamming64(h1, h2)
    dist_1_3 = hamming64(h1, h3)

    assert dist_1_2 < dist_1_3
    assert dist_1_2 <= 8
