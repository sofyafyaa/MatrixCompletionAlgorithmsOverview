from dataclasses import dataclass
from typing import List, Dict
import json
import os

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
    ranks: List[int]
    OSs: List[float]
    noise_level: float
    seed: int


def evaluate(evaluation_config: EvaluationConfig):
    MG = MatrixGenerator()
    test_cases = []
    for rank in evaluation_config.ranks:
        for OS in evaluation_config.OSs:
            M, Omega = MG.get_matrix(
                m=evaluation_config.m,
                n=evaluation_config.n,
                k=rank,
                missing_fraction=OS,
                noise_level=evaluation_config.noise_level,
                random_state=evaluation_config.seed
            )
            test_cases.append((M, Omega, rank, OS))
    
    for model in evaluation_config.models_evaluation_config:
        print(f'Evaluating {model.title}...')
        experiments = []
        for M, Omega, rank, OS in test_cases:
            print(f'rank={rank} OS={OS}...')
            model.model.complete_matrix(M=M, Omega=Omega, **model.kwargs)
            experiments.append({
                'iters_info': model.model.iters_info,
                'rank': rank,
                'OS': OS
            })
        path = os.path.join('plots', model.title + '.png')
        model.model.plot_info(path, experiments)


if __name__ == '__main__':
    alpha = 0.33 # Normalization parameter for RGD and RCG
    num_iters = 30_000 # Maximum number of iterations for completion
    tol = 1e-6 # Convergence tolerance, np.inf if it not need
    
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
        m=900,
        n=900,
        ranks=[10, 50],
        OSs=[0.7, 0.9],
        noise_level=0,
        seed=42
    )
    evaluate(evaluation_config=evaluation_config)
