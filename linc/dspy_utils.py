import os
import dspy
import argparse
import functools

from logic import prove, find_fol_errors, OWA_PRED
from dspy.datasets import DataLoader
from dspy.teleprompt import MIPRO, BootstrapFewShotWithRandomSearch
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    return parser.parse_args()

def load_proofwriter(dataset_name: str = "theoxo/proofwriter-deduction-balanced") -> DataLoader:
    return DataLoader().from_huggingface(dataset_name, input_keys=('theory', 'question'))

class FOLGen(dspy.Signature):
    """Convert the theory and question statements to First Order Logic (FOL) Expressions. 
    Expressions should be adhere to the format of the Python NLTK package logic module (e.g. "If Ohio is a State, Ohio is in the USA" becomes "State(Ohio) -> IsIn(Ohio, USA)"). 
    Each expression should be on its own line, with the question expression being the final line. 
    Do NOT wrap the FOL expressions in Markdown.
    Do NOT have extra text before the FOL expression.
    Do NOT use special characters like ∃ or ∀.
    Do NOT use strings like ~, <=>, or =>, use -, <->, and -> instead.
    Do NOT use function definitions of operations (And(), Or(), Not()), use symbols instead (&, |, -)
    Do NOT use symbols with multiple arities.
    Wrap all FOL expressions with <EVALUATE></EVALUATE> tags on their own line (<EVALUATE> is first line, </EVALUATE> is last line)"""

    theory = dspy.InputField()
    question = dspy.InputField()
    fol_statements = dspy.OutputField()

class FOLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(FOLGen)
    
    def forward(self, theory, question, **kwargs):
        response = self.prog(theory=theory, question=question)
        print(response)
        lines = [l.strip() for l in response.fol_statements.strip().split('\n')]
        lines = [l for l in lines if len(l) != 0]
        premises, conclusion = lines[1:-2], lines[-2]
        error = find_fol_errors(premises, conclusion)
        dspy.Suggest(
            error == "",
            msg = f"Encountered Logic Error when Parsing FOL Expressions:\n{error}"
        )
        return response

def fol_gen_metric(example, pred) -> int:
    try:
        # parse out lines
        lines = [l.strip() for l in pred.fol_statements.strip().split('\n')]
        # get rid of empty lines
        lines = [l for l in lines if len(l) != 0]
        # parse out terminal periods
        lines = [l[:-1] if l[-1] == '.' else l for l in lines]
        # yes I know this is hacky stfu
        # lines = [l.replace('~', '-') for l in lines]
        premises, conclusion = lines[1:-2], lines[-2]
        print("PREMISES:\n", premises)
        print("CONCLUSION:\n", conclusion)
        output = prove(premises, conclusion)
        print("OUTPUT:\n", output)

        match output:
            case OWA_PRED.TRUE:
                output_str = "True"
            case OWA_PRED.FALSE:
                output_str = "False"
            case OWA_PRED.UNK:
                output_str = "Uncertain"
            case OWA_PRED.ERR:
                output_str = "Error"
        
        return min(int(output_str == example.answer) + 0.5, 1)
    except Exception as e:
        return 0

if __name__ == "__main__":
    args = parse_args()
    if args.model == "":
        lm = dspy.Cohere(model="command-r-plus", api_key=os.getenv("COHERE_API_KEY"), max_tokens=10000, stop_sequences=["</EVALUATE>"])
    else:
        lm = dspy.HFModel(model=args.hf_model)

    dspy.configure(lm=lm)
    dspy.configure(trace=[])

    # predictor = dspy.Predict(FOLGen)
    predictor = FOLGenerator()
    predictor_with_assertions = assert_transform_module(predictor, assertion_handler=functools.partial(backtrack_handler, max_backtracks=2))
    dataset = load_proofwriter()
    trainset = dataset['train']
    # out = predictor(theory=dataset['train'][0]['theory'], question=dataset['train'][0]['question'])
    fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=fol_gen_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=1)
    your_dspy_program_compiled = fewshot_optimizer.compile(student = predictor_with_assertions, trainset=trainset)

    your_dspy_program_compiled.save('optimized_program.json')

    # num_new_prompts_generated = 10
    # prompt_generation_temperature = 0.8
    # teleprompter = MIPRO(prompt_model=lm, task_model=lm, metric=fol_gen_metric, n=num_new_prompts_generated, init_temperature=prompt_generation_temperature)
    # kwargs = dict(num_threads=1, display_progress=True, display_table=0)
    # = teleprompter.compile(predictor, trainset=dataset['train'], num_trials=100, max_bootstrapped_demos=3, max_labeled_demos=5, eval_kwargs=kwargs)

