from tensorly import decomposition
import tensorly

def tensor3_to_factors(tensor, rank, n_iter_max=50, init='svd', verbose=10, random_state=1):
    assert init in ['random', 'svd']
    factors, rec_errors = decomposition.parafac(tensor, rank=rank, random_state=random_state, normalize_factors=False, verbose=verbose,
                                    n_iter_max=n_iter_max, tol=1e-4, init=init, return_errors=True)

    return factors[1][0], factors[1][1], factors[1][2], rec_errors

def recover_tensor_from_factors(factors):

    recovered = tensorly.cp_to_tensor(factors)
    return recovered
