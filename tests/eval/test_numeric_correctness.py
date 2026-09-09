from eval.numeric_correctness import numeric_correctness


def test_numeric_correctness_supports_scalar_units_and_intermediate_values() -> None:
    assert (
        numeric_correctness(
            "Netflix's FY2016 EBIT is: $438.00 million",
            "Netflix's FY2016 EBIT is $438.00 million. The ratio is 2.92 times.",
            task_type="Implicit_Reasoning",
        )
        == 1.0
    )
    assert numeric_correctness("0.6116", "现金比率为61.16%", task_type="Implicit_Reasoning") == 1.0
    assert numeric_correctness("126.3816", "缺少相关数据，无法求出答案", task_type="Implicit_Reasoning") == 0.0


def test_numeric_correctness_supports_lists_and_tolerance() -> None:
    assert (
        numeric_correctness(
            "[2.81, 0.030, 8.706]",
            "[2.814, 0.0304, 8.7064]",
            task_type="Explicit_Reasoning",
        )
        == 1.0
    )
    assert numeric_correctness("34.6", "结果为34.64个百分点", task_type="Explicit_Reasoning") == 1.0
    assert numeric_correctness("34.6", "结果为34.66个百分点", task_type="Explicit_Reasoning") == 0.0


def test_numeric_correctness_matches_amounts_with_thousands_separators() -> None:
    reference = "工程服务年度上限为600,000万元，实际交易金额为495,112.84万元。"
    prediction = "工程服务的年度上限为600000万元，实际交易金额为495112.840万元，使用率为82.5%。"

    assert numeric_correctness(reference, prediction, task_type="Comparison") == 1.0


def test_numeric_correctness_is_not_applicable_to_narrative_answers() -> None:
    reference = "综合证据表明资本结构和盈利能力均相对稳定，利润率变化约为2个百分点。"

    assert numeric_correctness(reference, "答案中提到2个百分点。", task_type="MultiHop_Judgment") is None
