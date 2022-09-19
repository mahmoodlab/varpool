from var_pool.nn.train.tests.utils_grad_accum import check_GradAccum


def test_GradAccum():

    assert check_GradAccum(n_samples=5, batch_size=1, grad_accum=2)
    assert check_GradAccum(n_samples=5, batch_size=1, grad_accum=3)

    assert check_GradAccum(n_samples=4, batch_size=2, grad_accum=1)
    assert check_GradAccum(n_samples=4, batch_size=2, grad_accum=2)
    assert check_GradAccum(n_samples=5, batch_size=2, grad_accum=2)

    assert check_GradAccum(n_samples=10, batch_size=3, grad_accum=2)
    assert check_GradAccum(n_samples=10, batch_size=3, grad_accum=3)
    assert check_GradAccum(n_samples=10, batch_size=3, grad_accum=4)
