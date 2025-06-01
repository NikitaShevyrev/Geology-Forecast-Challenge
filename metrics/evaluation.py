import numpy as np
from scipy.special import logsumexp


def compute_nll_score(solution, submission, row_id_column_name='geology_id'):
    solution_copy = solution.copy()
    submission_copy = submission.copy()

    del solution_copy[row_id_column_name]
    del submission_copy[row_id_column_name]

    NEGATIVE_PART = -299
    LARGEST_CHUNK = 600
    SMALLEST_CHUNK = 350
    TOTAL_REALIZATIONS = 10
    INFLATION_SIGMA = 600
    
    sigma_2 = np.ones((LARGEST_CHUNK+NEGATIVE_PART-1))
    from_ranges = [1, 61, 245]
    to_ranges_excl = [61, 245, 301]
    log_slopes = [1.0406028049510443, 0.0, 7.835345062351012]
    log_offsets = [-6.430669850650689, -2.1617411566043896, -45.24876794412965]

    for growth_mode in range(len(from_ranges)):
        for i in range(from_ranges[growth_mode], to_ranges_excl[growth_mode]):
            sigma_2[i-1] = np.exp(np.log(i)*log_slopes[growth_mode]+log_offsets[growth_mode])

    sigma_2 *= INFLATION_SIGMA
  
    cov_matrix_inv_diag = 1. / sigma_2
    
    num_rows = solution_copy.shape[0]
    num_columns = LARGEST_CHUNK + NEGATIVE_PART - 1
    
    p = 1./TOTAL_REALIZATIONS
    log_p = np.log(p)
    
    solution_arr = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))
    submission_arr = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))
    
    for k in range(TOTAL_REALIZATIONS):
        for i in range(num_columns):
            column_name = f"r_{k}_pos_{i+1}"
            solution_arr[:, k, i] = solution_copy[column_name].values
            submission_arr[:, k, i] = submission_copy[column_name].values

    misfit = solution_arr - submission_arr
    inner_product_matrix = np.sum(cov_matrix_inv_diag * misfit * misfit, axis=2)
    
    nll = -logsumexp(log_p - inner_product_matrix, axis=1)
    
    return nll.mean()