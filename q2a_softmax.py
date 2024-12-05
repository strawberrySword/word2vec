import numpy as np

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    # Check for an empty array
    if x.size == 0:
        return x

    if len(x.shape) > 1:
        x = np.apply_along_axis(softmax, 1, x)

    else:
        x -= np.max(x) # Prevent overflow
        denom = np.sum(np.exp(x))
        x = (np.exp(x))/denom

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")

    # Test with large numbers
    test1 = softmax(np.array([1000, 1000, 1000]))
    print("Large numbers test:", test1)
    ans1 = np.array([1/3, 1/3, 1/3])  # Softmax of identical values
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    # Test with very small numbers
    test2 = softmax(np.array([-1000, -1000, -1000]))
    print("Small numbers test:", test2)
    ans2 = np.array([1/3, 1/3, 1/3])  # Softmax of identical values
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    # Test with a mix of very large and very small numbers
    test3 = softmax(np.array([1000, -1000]))
    print("Mixed large and small numbers test:", test3)
    ans3 = np.array([1.0, 0.0])  # Only the largest value should dominate
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    # Test with a single value
    test4 = softmax(np.array([0]))
    print("Single value test:", test4)
    ans4 = np.array([1.0])  # Only one value, so it gets probability 1
    assert np.allclose(test4, ans4, rtol=1e-05, atol=1e-06)

    # Test with a 2D array with zeroes
    test5 = softmax(np.array([[0, 0, 0], [0, 0, 0]]))
    print("2D array with zeroes test:", test5)
    ans5 = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])  # Uniform distribution
    assert np.allclose(test5, ans5, rtol=1e-05, atol=1e-06)

    # Test with a 2D array with different ranges
    test6 = softmax(np.array([[1, 2, 3], [-1, -2, -3]]))
    print("2D array with different ranges test:", test6)
    ans6 = np.array([
        [0.09003057, 0.24472847, 0.66524096],
        [0.66524096, 0.24472847, 0.09003057]])
    assert np.allclose(test6, ans6, rtol=1e-05, atol=1e-06)

    # Test with an empty array
    try:
        test7 = softmax(np.array([]))
        print("Empty array test:", test7)
        assert test7.size == 0  # Should return an empty array
    except Exception as e:
        print("Empty array test failed with exception:", e)

    print("All edge case tests passed!\n")


if __name__ == "__main__":
    test_softmax_basic()
    your_softmax_test()
