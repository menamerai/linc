import re


def convert_to_nltk_rep(logic_formula):
    # THIS IS A DIRECT COPY FROM THE LINC REPO, REMOVE THIS CODE WHEN SUBMIT
    translation_map = {
        "∀": "all ",
        "∃": "exists ",
        "→": "->",
        "¬": "-",
        "∧": "&",
        "∨": "|",
        "⟷": "<->",
        "↔": "<->",
        "0": "Zero",
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
        ".": "Dot",
        "Ś": "S",
        "ą": "a",
        "’": "",
    }

    constant_pattern = r"\b([a-z]{2,})(?!\()"
    logic_formula = re.sub(
        constant_pattern, lambda match: match.group(1).capitalize(), logic_formula
    )

    for key, value in translation_map.items():
        logic_formula = logic_formula.replace(key, value)

    quant_pattern = r"(all\s|exists\s)([a-z])"

    def replace_quant(match):
        return match.group(1) + match.group(2) + "."

    logic_formula = re.sub(quant_pattern, replace_quant, logic_formula)

    dotted_param_pattern = r"([a-z])\.(?=[a-z])"

    def replace_dotted_param(match):
        return match.group(1)

    logic_formula = re.sub(dotted_param_pattern, replace_dotted_param, logic_formula)

    simple_xor_pattern = r"(\w+\([^()]*\)) ⊕ (\w+\([^()]*\))"

    def replace_simple_xor(match):
        return (
            "(("
            + match.group(1)
            + " & -"
            + match.group(2)
            + ") | (-"
            + match.group(1)
            + " & "
            + match.group(2)
            + "))"
        )

    logic_formula = re.sub(simple_xor_pattern, replace_simple_xor, logic_formula)

    complex_xor_pattern = r"\((.*?)\)\) ⊕ \((.*?)\)\)"

    def replace_complex_xor(match):
        return (
            "((("
            + match.group(1)
            + ")) & -("
            + match.group(2)
            + "))) | (-("
            + match.group(1)
            + ")) & ("
            + match.group(2)
            + "))))"
        )

    logic_formula = re.sub(complex_xor_pattern, replace_complex_xor, logic_formula)

    special_xor_pattern = r"\(\(\((.*?)\)\)\) ⊕ (\w+\([^()]*\))"

    def replace_special_xor(match):
        return (
            "((("
            + match.group(1)
            + ")) & -"
            + match.group(2)
            + ") | (-("
            + match.group(1)
            + ")) & "
            + match.group(2)
            + ")"
        )

    logic_formula = re.sub(special_xor_pattern, replace_special_xor, logic_formula)

    return logic_formula
