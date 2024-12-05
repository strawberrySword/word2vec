import random

import numpy as np
from numpy.testing import assert_allclose


def gradcheck_naive(f, x, gradient_text=""):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradient_text -- a string detailing some context about the gradient computation
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4         # Do not change this!

    # Valid value check
    if np.any(np.isinf(x)) or np.any(np.isnan(x)):
        raise ValueError(f"Invalid input detected: {x}. Gradient check is not valid for infinities or NaNs.")

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Save the original value of i-th demention
        original_xi_value = x[ix]

        # Modify the i-th dimention to get f(x + h)
        x[ix] = original_xi_value + h
        random.setstate(rndstate)  # Ensure consistent results
        fx_plus_h, _ = f(x)

        # Modify the i-th dimention to get f(x - h)
        x[ix] = original_xi_value - h
        random.setstate(rndstate)  # Ensure consistent results
        fx_minus_h, _ = f(x)

        # Restore the original value
        x[ix] = original_xi_value

        # Compute the numerical gradient
        numgrad = (fx_plus_h - fx_minus_h) / (2 * h)

        # Compare gradients
        assert_allclose(numgrad, grad[ix], rtol=1e-5,
                        err_msg=f"Gradient check failed for {gradient_text}.\n"
                                f"First gradient error found at index {ix} in the vector of gradients\n"
                                f"Your gradient: {grad[ix]} \t Numerical gradient: {numgrad}")

        it.iternext()  # Step to next dimension

    print("Gradient check passed!")


def test_gradcheck_basic():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), 2*x)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))       # scalar test
    gradcheck_naive(quad, np.random.randn(3,))     # 1-D test
    gradcheck_naive(quad, np.random.randn(4, 5))   # 2-D test
    print()


def your_gradcheck_test():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    
    # Simple linear function: gradcheck should pass
    linear = lambda x: (np.sum(x), np.ones_like(x))
    gradcheck_naive(linear, np.zeros(5), gradient_text="Simple linear function (zero vector)")  # Zero vector
    gradcheck_naive(linear, np.ones(5), gradient_text="Simple linear function (ones vector)")  # Ones vector

    # Test with large values
    large_values = lambda x: (np.sum(x ** 2), 2 * x)
    gradcheck_naive(large_values, np.full((5,), 1e5), gradient_text=" Large values (1e5)")

    # Test with very small values
    small_values = lambda x: (np.sum(x**2), 2*x)
    gradcheck_naive(small_values, np.full((5,), 1e-10), gradient_text="Small values (1e-10)")

    # Test with high-dimensional array
    high_dim = lambda x: (np.sum(x ** 2), 2 * x)
    gradcheck_naive(high_dim, np.random.randn(10, 10, 10), gradient_text="High-dimensional array")

    print("Edge case tests passed!")


if __name__ == "__main__":
    test_gradcheck_basic()
    your_gradcheck_test()
