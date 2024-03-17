import re

from nltk.inference import Prover9
from nltk.sem import Expression

from typing import List
from pred_types import OWA_PRED

# hack to get the parsing function into a cleaner form
read_expr = Expression.fromstring
prover = Prover9()

def get_all_variables(s: str) -> List[str]:
    pattern = "\([^()]+\)"
    matches = re.findall(pattern, s)
    all_vars = set()
    for m in matches:
        m = m[1:-1]
        subs = m.split(',')
        all_vars.update([s.strip() for s in subs])
    return list(all_vars)

def format_fol(s: str) -> str:
    reps = {
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
        "'": "",
        "-": "_",
        "`": "",
        " ": "_"
    }

    vars = get_all_variables(s)
    for var in vars:
        new_var = var[:]
        for k, v in reps.items():
            new_var = new_var.replace(k, v)
        s = s.replace(var, new_var)
    return s 

def convert_to_nltk_rep(logic_formula):
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

    constant_pattern = r'\b([a-z]{2,})(?!\()'
    logic_formula = re.sub(constant_pattern, lambda match: match.group(1).capitalize(), logic_formula)

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
        return ("((" + match.group(1) + " & -" + match.group(2) + ") | (-" + match.group(1) + " & " + match.group(2) + "))")
    logic_formula = re.sub(simple_xor_pattern, replace_simple_xor, logic_formula)

    complex_xor_pattern = r"\((.*?)\)\) ⊕ \((.*?)\)\)"
    def replace_complex_xor(match):
        return ("(((" + match.group(1) + ")) & -(" + match.group(2) + "))) | (-(" + match.group(1) + ")) & (" + match.group(2) + "))))")
    logic_formula = re.sub(complex_xor_pattern, replace_complex_xor, logic_formula)

    special_xor_pattern = r"\(\(\((.*?)\)\)\) ⊕ (\w+\([^()]*\))"
    def replace_special_xor(match):
        return ("(((" + match.group(1) + ")) & -" + match.group(2) + ") | (-(" + match.group(1) + ")) & " + match.group(2) + ")")
    logic_formula = re.sub(special_xor_pattern, replace_special_xor, logic_formula)
    
    return logic_formula

def prove(premises: List[str], conclusion: str) -> OWA_PRED:
    prem_exprs = [read_expr(format_fol(p)) for p in premises]
    conc_expr = read_expr(format_fol(conclusion))

    conc_provable = prover.prove(conc_expr, prem_exprs)
    print(conc_provable)
    # this is a bit of a hack, but use short circuit eval to avoid second proving if possible
    conc_deniable = (not conc_provable) and prover.prove(conc_expr.negate(), prem_exprs)
    print(conc_deniable)
    if conc_provable:
        return OWA_PRED.TRUE
    elif conc_deniable:
        return OWA_PRED.FALSE
    else:
        return OWA_PRED.UNK