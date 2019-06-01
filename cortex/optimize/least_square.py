import cvxpy as cp

def solve_least_square(A, b, min_constraint=None, max_constraint=None):
    '''
    Use CVXPY for optimizing least square problems with box constraints
    :param A: Normal matrix
    :param b: residuals
    :param min_constraint:
    :param max_constraint:
    :return: the least square solution that minimizes Ax-b
    '''

    m, n = A.shape
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A * x - b))
    constraints = []
    if min_constraint is not None:
        constraints.append(min_constraint <= x)
    if max_constraint is not None:
        constraints.append(x <= max_constraint)

    prob = cp.Problem(objective, constraints)

    prob = cp.Problem(objective)
    result = prob.solve()

    # The optimal Lagrange multiplier for a constraint is stored in constraint.dual_value.
    # constraints[0].dual_value
    return x.value,


