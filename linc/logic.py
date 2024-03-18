import re
from typing import List

from custom_types import OWA_PRED
from nltk.inference import Prover9
from nltk.inference.prover9 import Prover9FatalException
from nltk.sem import Expression
from utils import convert_to_nltk_rep

# hack to get the parsing function into a cleaner form
read_expr = Expression.fromstring
prover = Prover9()


def get_all_variables(s: str) -> List[str]:
    pattern = "\([^()]+\)"
    matches = re.findall(pattern, s)
    all_vars = set()
    for m in matches:
        m = m[1:-1]
        subs = m.split(",")
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
        " ": "_",
    }

    vars = get_all_variables(s)
    for var in vars:
        new_var = var[:]
        for k, v in reps.items():
            new_var = new_var.replace(k, v)
        s = s.replace(var, new_var)
    return s


def prove(premises: List[str], conclusion: str) -> OWA_PRED:
    # format for NLTK just bc
    # premises = [convert_to_nltk_rep(p) for p in premises]
    # conclusion = convert_to_nltk_rep(conclusion)

    # parse the expressions into nltk Expression objects
    prem_exprs = [read_expr(format_fol(p)) for p in premises]
    conc_expr = read_expr(format_fol(conclusion))
    print("PARSED CORRECTLY")

    # attempt to prove whether the conclusion is true from the premises
    try:
        conc_provable = prover.prove(conc_expr, prem_exprs)
    except Prover9FatalException as e:
        print(e)
        return OWA_PRED.ERR

    # attempt to prove whether the conclusion is deniable from the premises
    # this is a bit of a hack, but use short circuit eval to avoid second proving if possible
    try:
        conc_deniable = (not conc_provable) and prover.prove(
            conc_expr.negate(), prem_exprs
        )
    except Prover9FatalException as e:
        print("SECOND ERROR")
        print(e)
        return OWA_PRED.ERR

    # return the according OWA flag
    if conc_provable:
        return OWA_PRED.TRUE
    elif conc_deniable:
        return OWA_PRED.FALSE
    else:
        return OWA_PRED.UNK
