from dataclasses import dataclass
from typing import List, Dict
import time
import json

from MatrixCompletionClass import MatrixCompletion
from Dataset import MatrixGenerator
from algorithms.RCGMatrixCompletion import RCGMatrixCompletion


@dataclass
class ModelEvaluationConfig:
    title: str
    model: MatrixCompletion
    kwargs: Dict
    

@dataclass
class EvaluationConfig:
    models_evaluation_config: List[ModelEvaluationConfig]
    m: int
    n: int
    rank: int
    OS: float
    noise_level: float
    seed: int


def evaluate(evaluation_config: EvaluationConfig):
    MG = MatrixGenerator()
    M, Omega = MG.get_matrix(
        m=evaluation_config.m,
        n=evaluation_config.n,
        k=evaluation_config.rank,
        missing_fraction=evaluation_config.OS,
        noise_level=evaluation_config.noise_level,
        random_state=evaluation_config.seed
    )
    
    for model in evaluation_config.models_evaluation_config:
        print(f'Evaluating {model.title}...')
        start = time.time()
        model.model.complete_matrix(M=M, Omega=Omega, **model.kwargs)
        finish = time.time()
        elapsed_time = finish - start
        print('Elapsed time: ', elapsed_time)
        model.model.plot_info()


if __name__ == '__main__':
    alpha = 0.33 # Normalization parameter for RGD and RCG
    num_iters = 1_000 # Maximum number of iterations for completion
    tol = 1e-3 # Convergence tolerance, np.inf if it not need
    
    rcg_rgd_model_config = json.dumps({
        "alpha": alpha,
        "num_iters": num_iters,
        "tol": tol,
    })
    
    models_evaluation_config = [
        ModelEvaluationConfig(
            title=f'RGD (QPRECON) alpha={alpha}',
            model=RCGMatrixCompletion(params_str=rcg_rgd_model_config),
            kwargs={'method': 'rgd', 'metric': 'QPRECON'}
        ),
        ModelEvaluationConfig(
            title=f'RGD (QRIGHT-INV) alpha={alpha}',
            model=RCGMatrixCompletion(params_str=rcg_rgd_model_config),
            kwargs={'method': 'rgd', 'metric': 'QRIGHT-INV'}
        ),
        ModelEvaluationConfig(
            title=f'RCG (QPRECON) alpha={alpha}',
            model=RCGMatrixCompletion(params_str=rcg_rgd_model_config),
            kwargs={'method': 'rcg', 'metric': 'QPRECON'}
        ),
        ModelEvaluationConfig(
            title=f'RCG (QRIGHT-INV) alpha={alpha}',
            model=RCGMatrixCompletion(params_str=rcg_rgd_model_config),
            kwargs={'method': 'rcg', 'metric': 'QRIGHT-INV'}
        ),
    ]
    evaluation_config = EvaluationConfig(
        models_evaluation_config=models_evaluation_config,
        m=50,
        n=50,
        rank=10,
        OS=0.2,
        noise_level=0,
        seed=42
    )
    evaluate(evaluation_config=evaluation_config)
