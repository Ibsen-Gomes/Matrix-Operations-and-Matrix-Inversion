###################################################### Ibsen P. S. Gomes ############################################################
########################################## Mátodos Numéricos - Obseratório Nacional #################################################
#################################################### Template de testes #############################################################

# Bibliotecas:
import numpy as np
import scipy as sp
from numpy.testing import assert_almost_equal as aae
import pytest
import template_ibsen as temp

from scipy.linalg import dft
from scipy.fft import fft, ifft # funções para série de Fourier
from scipy.linalg import lu # função do python para obter LU, para fins de comparação 
from scipy.linalg import toeplitz, circulant

### scalar-vector product

def test_scalar_vec_real_dumb_a_not_scalar():
    'fail if a is not a scalar'
    # 2d array
    a1 = np.ones((3,2))
    # list
    a2 = [7.]
    # tuple
    a3 = (4, 8.2)
    vector = np.arange(4)
    for ai in [a1, a2, a3]:
        with pytest.raises(AssertionError):
            temp.scalar_vec_real_dumb(ai, vector)


def test_scalar_vec_real_numpy_a_not_scalar():
    'fail if a is not a scalar'
    # 2d array
    a1 = np.ones((3,2))
    # list
    a2 = [7.]
    # tuple
    a3 = (4, 8.2)
    vector = np.arange(4)
    for ai in [a1, a2, a3]:
        with pytest.raises(AssertionError):
            temp.scalar_vec_real_numpy(ai, vector)


def test_scalar_vec_real_numba_a_not_scalar():
    'fail if a is not a scalar'
    # 2d array
    a1 = np.ones((3,2))
    # list
    a2 = [7.]
    # tuple
    a3 = (4, 8.2)
    vector = np.arange(4)
    for ai in [a1, a2, a3]:
        with pytest.raises(AssertionError):
            temp.scalar_vec_real_numba(ai, vector)


def test_scalar_vec_real_x_not_1darray():
    'fail if x is not a 1d array'
    a = 2
    # 2d array
    x1 = np.ones((3,2))
    # string
    x2 = 'not array'
    for xi in [x1, x2]:
        with pytest.raises(AssertionError):
            temp.scalar_vec_real_dumb(a, xi)
    for xi in [x1, x2]:
        with pytest.raises(AssertionError):
            temp.scalar_vec_real_numpy(a, xi)
    for xi in [x1, x2]:
        with pytest.raises(AssertionError):
            temp.scalar_vec_real_numba(a, xi)


def test_scalar_vec_real_known_values():
    'check output produced by specific input'
    scalar = 1
    vector = np.linspace(23.1, 52, 10)
    reference_output = np.copy(vector)
    computed_output_dumb = temp.scalar_vec_real_dumb(scalar, vector)
    computed_output_numpy = temp.scalar_vec_real_numpy(scalar, vector)
    computed_output_numba = temp.scalar_vec_real_numba(scalar, vector)
    aae(reference_output, computed_output_dumb, decimal=10)
    aae(reference_output, computed_output_numpy, decimal=10)
    aae(reference_output, computed_output_numba, decimal=10)


def test_scalar_vec_real_ignore_complex():
    'complex part of input must be ignored'
    scalar = 3.
    vector = np.ones(4) + 1j*np.ones(4)
    reference_output = np.zeros(4) + 3.
    computed_output_dumb = temp.scalar_vec_real_dumb(scalar, vector)
    computed_output_numpy = temp.scalar_vec_real_numpy(scalar, vector)
    computed_output_numba = temp.scalar_vec_real_numba(scalar, vector)
    aae(reference_output, computed_output_dumb, decimal=10)
    aae(reference_output, computed_output_numpy, decimal=10)
    aae(reference_output, computed_output_numba, decimal=10)


def test_scalar_vec_complex_functions_compare_numpy():
    'compare scalar_vec_complex dumb, numpy and numba with numpy'
    np.random.seed(3)
    scalar = np.random.rand() + 1j*np.random.rand()
    vector = np.random.rand(13) + np.random.rand(13)*1j
    output_dumb = temp.scalar_vec_complex(scalar, vector, function='dumb')
    output_numpy = temp.scalar_vec_complex(scalar, vector, function='numpy')
    output_numba = temp.scalar_vec_complex(scalar, vector, function='numba')
    reference = scalar*vector
    aae(output_dumb, reference, decimal=10)
    aae(output_numpy, reference, decimal=10)
    aae(output_numba, reference, decimal=10)


### dot_product

def test_dot_real_not_1D_arrays():
    'fail due to input that is not 1D array'
    vector_1 = np.ones((3,2))
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        temp.dot_real_dumb(vector_1, vector_2)
    with pytest.raises(AssertionError):
        temp.dot_real_numpy(vector_1, vector_2)
    with pytest.raises(AssertionError):
        temp.dot_real_numba(vector_1, vector_2)


def test_dot_real_different_sizes():
    'fail due to inputs having different sizes'
    vector_1 = np.linspace(5,6,7)
    vector_2 = np.arange(4)
    with pytest.raises(AssertionError):
        temp.dot_real_dumb(vector_1, vector_2)
    with pytest.raises(AssertionError):
        temp.dot_real_numpy(vector_1, vector_2)
    with pytest.raises(AssertionError):
        temp.dot_real_numba(vector_1, vector_2)


def test_dot_real_known_values():
    'check output produced by specific input'
    vector_1 = 0.1*np.ones(10)
    vector_2 = np.linspace(23.1, 52, 10)
    reference_output = np.mean(vector_2)
    computed_output_dumb = temp.dot_real_dumb(vector_1, vector_2)
    computed_output_numpy = temp.dot_real_numpy(vector_1, vector_2)
    computed_output_numba = temp.dot_real_numba(vector_1, vector_2)
    aae(reference_output, computed_output_dumb, decimal=10)
    aae(reference_output, computed_output_numpy, decimal=10)
    aae(reference_output, computed_output_numba, decimal=10)


def test_dot_real_compare_numpy_dot():
    'compare with numpy.dot'
    np.random.seed(41)
    vector_1 = np.random.rand(13)
    vector_2 = np.random.rand(13)
    reference_output_numpy = np.dot(vector_1, vector_2)
    computed_output_dumb = temp.dot_real_dumb(vector_1, vector_2)
    computed_output_numpy = temp.dot_real_numpy(vector_1, vector_2)
    computed_output_numba = temp.dot_real_numba(vector_1, vector_2)
    aae(reference_output_numpy, computed_output_dumb, decimal=10)
    aae(reference_output_numpy, computed_output_numpy, decimal=10)
    aae(reference_output_numpy, computed_output_numba, decimal=10)


def test_dot_real_commutativity():
    'verify commutativity'
    np.random.seed(19)
    a = np.random.rand(15)
    b = np.random.rand(15)
    # a dot b = b dot a
    output_ab_dumb = temp.dot_real_dumb(a, b)
    output_ba_dumb = temp.dot_real_dumb(b, a)
    output_ab_numpy = temp.dot_real_numpy(a, b)
    output_ba_numpy = temp.dot_real_numpy(b, a)
    output_ab_numba = temp.dot_real_numba(a, b)
    output_ba_numba = temp.dot_real_numba(b, a)
    aae(output_ab_dumb, output_ba_dumb, decimal=10)
    aae(output_ab_numpy, output_ba_numpy, decimal=10)
    aae(output_ab_numba, output_ba_numba, decimal=10)


def test_dot_real_distributivity():
    'verify distributivity over sum'
    np.random.seed(5)
    a = np.random.rand(15)
    b = np.random.rand(15)
    c = np.random.rand(15)
    # a dot (b + c) = (a dot b) + (a dot c)
    output_a_bc_dumb = temp.dot_real_dumb(a, b + c)
    output_ab_ac_dumb = temp.dot_real_dumb(a, b) + temp.dot_real_dumb(a, c)
    output_a_bc_numpy = temp.dot_real_numpy(a, b + c)
    output_ab_ac_numpy = temp.dot_real_numpy(a, b) + temp.dot_real_numpy(a, c)
    output_a_bc_numba = temp.dot_real_numba(a, b + c)
    output_ab_ac_numba = temp.dot_real_numba(a, b) + temp.dot_real_numba(a, c)
    aae(output_a_bc_dumb, output_ab_ac_dumb, decimal=10)
    aae(output_a_bc_numpy, output_ab_ac_numpy, decimal=10)
    aae(output_a_bc_numba, output_ab_ac_numba, decimal=10)


def test_dot_real_scalar_multiplication():
    'verify scalar multiplication property'
    np.random.seed(8)
    a = np.random.rand(15)
    b = np.random.rand(15)
    c1 = 5.6
    c2 = 9.1
    # (c1 a) dot (c2 b) = c1c2 (a dot b)
    output_c1a_c2b_dumb = temp.dot_real_dumb(c1*a, c2*b)
    output_c1c2_ab_dumb = c1*c2*temp.dot_real_dumb(a, b)
    output_c1a_c2b_numpy = temp.dot_real_numpy(c1*a, c2*b)
    output_c1c2_ab_numpy = c1*c2*temp.dot_real_numpy(a, b)
    output_c1a_c2b_numba = temp.dot_real_numba(c1*a, c2*b)
    output_c1c2_ab_numba = c1*c2*temp.dot_real_numba(a, b)
    aae(output_c1a_c2b_dumb, output_c1c2_ab_dumb, decimal=10)
    aae(output_c1a_c2b_numpy, output_c1c2_ab_numpy, decimal=10)
    aae(output_c1a_c2b_numba, output_c1c2_ab_numba, decimal=10)


def test_dot_real_ignore_complex():
    'complex part of input must be ignored'
    vector_1 = 0.1*np.ones(10)
    vector_2 = np.linspace(23.1, 52, 10) - 1j*np.ones(10)
    reference_output = np.mean(vector_2.real)
    computed_output_dumb = temp.dot_real_dumb(vector_1, vector_2)
    computed_output_numpy = temp.dot_real_numpy(vector_1, vector_2)
    computed_output_numba = temp.dot_real_numba(vector_1, vector_2)
    aae(reference_output, computed_output_dumb, decimal=10)
    aae(reference_output, computed_output_numpy, decimal=10)
    aae(reference_output, computed_output_numba, decimal=10)


def test_dot_complex_functions_compare_numpy_dot():
    'compare dot_complex_dumb, numpy and numba with numpy.dot'
    # first input complex
    np.random.seed(3)
    vector_1 = np.random.rand(13) + np.random.rand(13)*1j
    vector_2 = np.random.rand(13) + np.random.rand(13)*1j
    output_dumb = temp.dot_complex_dumb(vector_1, vector_2)
    output_numpy = temp.dot_complex_numpy(vector_1, vector_2)
    output_numba = temp.dot_complex_numba(vector_1, vector_2)
    output_numpy_dot = np.dot(vector_1, vector_2)
    aae(output_dumb, output_numpy_dot, decimal=10)
    aae(output_numpy, output_numpy_dot, decimal=10)
    aae(output_numba, output_numpy_dot, decimal=10)


def test_dot_complex_compare_numpy_dot():
    'compare dot_complex with numpy.dot'
    # first input complex
    np.random.seed(78)
    vector_1 = np.random.rand(10) + np.random.rand(10)*1j
    vector_2 = np.random.rand(10) + np.random.rand(10)*1j
    output_dumb = temp.dot_complex(vector_1, vector_2, function='dumb')
    output_numpy = temp.dot_complex(vector_1, vector_2, function='numpy')
    output_numba = temp.dot_complex(vector_1, vector_2, function='numba')
    output_numpy_dot = np.dot(vector_1, vector_2)
    aae(output_dumb, output_numpy_dot, decimal=10)
    aae(output_numpy, output_numpy_dot, decimal=10)
    aae(output_numba, output_numpy_dot, decimal=10)


def test_dot_complex_compare_numpy_vdot():
    'compare dot_complex with numpy.vdot'
    # first input complex
    np.random.seed(8)
    vector_1 = np.random.rand(10) + np.random.rand(10)*1j
    vector_2 = np.random.rand(10) + np.random.rand(10)*1j
    output_dumb = temp.dot_complex(vector_1, vector_2,
                                  conjugate=True, function='dumb')
    output_numpy = temp.dot_complex(vector_1, vector_2,
                                   conjugate=True, function='numpy')
    output_numba = temp.dot_complex(vector_1, vector_2,
                                   conjugate=True, function='numba')
    output_numpy_dot = np.vdot(vector_1, vector_2) # .conj()
    aae(output_dumb, output_numpy_dot, decimal=10)
    aae(output_numpy, output_numpy_dot, decimal=10)
    aae(output_numba, output_numpy_dot, decimal=10)


def test_dot_complex_invalid_function():
    'fail due to invalid function'
    vector_1 = np.ones(10)
    vector_2 = np.arange(10)+1.5
    with pytest.raises(ValueError):
        temp.dot_complex(vector_1, vector_2, function='not_valid_function')


# Hadamard product

def test_hadamard_real_different_shapes():
    'fail due to inputs having different sizes'
    a = np.linspace(5,10,8)
    B = np.ones((4,4))
    with pytest.raises(AssertionError):
        temp.hadamard_real_dumb(a, B)
    with pytest.raises(AssertionError):
        temp.hadamard_real_numpy(a, B)
    with pytest.raises(AssertionError):
        temp.hadamard_real_numba(a, B)


def test_hadamard_real_compare_asterisk():
    'compare hadamard_real function with * operator'
    # for vectors
    np.random.seed(7)
    input1 = np.random.rand(10)
    input2 = np.random.rand(10)
    output_dumb = temp.hadamard_real_dumb(input1, input2)
    output_numpy = temp.hadamard_real_numpy(input1, input2)
    output_numba = temp.hadamard_real_numba(input1, input2)
    output_asterisk = input1*input2
    aae(output_dumb, output_asterisk, decimal=10)
    aae(output_numpy, output_asterisk, decimal=10)
    aae(output_numba, output_asterisk, decimal=10)
    # for matrices
    np.random.seed(9)
    input1 = np.random.rand(5, 7)
    input2 = np.random.rand(5, 7)
    output_dumb = temp.hadamard_real_dumb(input1, input2)
    output_numpy = temp.hadamard_real_numpy(input1, input2)
    output_numba = temp.hadamard_real_numba(input1, input2)
    output_asterisk = input1*input2
    aae(output_dumb, output_asterisk, decimal=10)
    aae(output_numpy, output_asterisk, decimal=10)
    aae(output_numba, output_asterisk, decimal=10)


def test_hadamard_real_ignore_complex():
    'complex part of input must be ignored'
    # for vectors
    np.random.seed(7)
    input1 = np.random.rand(10)
    input2 = np.random.rand(10) + 1j*np.ones(10)
    output_dumb = temp.hadamard_real_dumb(input1, input2)
    output_numpy = temp.hadamard_real_numpy(input1, input2)
    output_numba = temp.hadamard_real_numba(input1, input2)
    output_reference = input1.real*input2.real
    aae(output_dumb, output_reference, decimal=10)
    aae(output_numpy, output_reference, decimal=10)
    aae(output_numba, output_reference, decimal=10)
    # for matrices
    np.random.seed(9)
    input1 = np.random.rand(5, 7) - 1j*np.ones((5,7))
    input2 = np.random.rand(5, 7)
    output_dumb = temp.hadamard_real_dumb(input1, input2)
    output_numpy = temp.hadamard_real_numpy(input1, input2)
    output_numba = temp.hadamard_real_numba(input1, input2)
    output_reference = input1.real*input2.real
    aae(output_dumb, output_reference, decimal=10)
    aae(output_numpy, output_reference, decimal=10)
    aae(output_numba, output_reference, decimal=10)


def test_hadamard_complex_compare_asterisk():
    'compare hadamard_complex function with * operator'
    # for matrices
    np.random.seed(34)
    input1 = np.random.rand(4, 3)
    input2 = np.random.rand(4, 3)
    output_dumb = temp.hadamard_complex(input1, input2, function='dumb')
    output_numpy = temp.hadamard_complex(input1, input2, function='numpy')
    output_numba = temp.hadamard_complex(input1, input2, function='numba')
    output_asterisk = input1*input2
    aae(output_dumb, output_asterisk, decimal=10)
    aae(output_numpy, output_asterisk, decimal=10)
    aae(output_numba, output_asterisk, decimal=10)


def test_hadamard_complex_invalid_function():
    'fail due to invalid function'
    vector_1 = np.ones(8)
    vector_2 = np.arange(8)+1.5
    with pytest.raises(ValueError):
        temp.hadamard_complex(vector_1, vector_2, function='not_valid_function')

# Outer product

def test_outer_real_input_not_vector():
    'fail with non-vector inputs'
    a = np.linspace(5,10,8)
    B = np.ones((4,4))
    with pytest.raises(AssertionError):
        temp.outer_real_dumb(a, B)
    with pytest.raises(AssertionError):
        temp.outer_real_numpy(a, B)
    with pytest.raises(AssertionError):
        temp.outer_real_numba(a, B)


def test_outer_real_compare_numpy_outer():
    'compare with numpy.outer'
    np.random.seed(301)
    vector_1 = np.random.rand(13)
    vector_2 = np.random.rand(13)
    reference_output_numpy = np.outer(vector_1, vector_2)
    computed_output_dumb = temp.outer_real_dumb(vector_1, vector_2)
    computed_output_numpy = temp.outer_real_numpy(vector_1, vector_2)
    computed_output_numba = temp.outer_real_numba(vector_1, vector_2)
    aae(reference_output_numpy, computed_output_dumb, decimal=10)
    aae(reference_output_numpy, computed_output_numpy, decimal=10)
    aae(reference_output_numpy, computed_output_numba, decimal=10)


def test_outer_real_known_values():
    'check output produced by specific input'
    vector_1 = np.ones(5)
    vector_2 = np.arange(1,11)
    reference_output = np.resize(vector_2, (vector_1.size, vector_2.size))
    computed_output_dumb = temp.outer_real_dumb(vector_1, vector_2)
    computed_output_numpy = temp.outer_real_numpy(vector_1, vector_2)
    computed_output_numba = temp.outer_real_numba(vector_1, vector_2)
    aae(reference_output, computed_output_dumb, decimal=10)
    aae(reference_output, computed_output_numpy, decimal=10)
    aae(reference_output, computed_output_numba, decimal=10)


def test_outer_real_transposition():
    'verify the transposition property'
    np.random.seed(72)
    a = np.random.rand(8)
    b = np.random.rand(5)
    a_outer_b_T_dumb = temp.outer_real_dumb(a, b).T
    b_outer_a_dumb = temp.outer_real_dumb(b, a)
    a_outer_b_T_numpy = temp.outer_real_numpy(a, b).T
    b_outer_a_numpy = temp.outer_real_numpy(b, a)
    a_outer_b_T_numba = temp.outer_real_numba(a, b).T
    b_outer_a_numba = temp.outer_real_numba(b, a)
    aae(a_outer_b_T_dumb, b_outer_a_dumb, decimal=10)
    aae(a_outer_b_T_numpy, b_outer_a_numpy, decimal=10)
    aae(a_outer_b_T_numba, b_outer_a_numba, decimal=10)


def test_outer_real_distributivity():
    'verify the distributivity property'
    np.random.seed(2)
    a = np.random.rand(5)
    b = np.random.rand(5)
    c = np.random.rand(4)
    a_plus_b_outer_c_dumb = temp.outer_real_dumb(a+b, c)
    a_outer_c_plus_b_outer_c_dumb = temp.outer_real_dumb(a, c) + \
                                    temp.outer_real_dumb(b, c)
    a_plus_b_outer_c_numpy = temp.outer_real_numpy(a+b, c)
    a_outer_c_plus_b_outer_c_numpy = temp.outer_real_numpy(a, c) + \
                                     temp.outer_real_numpy(b, c)
    a_plus_b_outer_c_numba = temp.outer_real_numba(a+b, c)
    a_outer_c_plus_b_outer_c_numba = temp.outer_real_numba(a, c) + \
                                     temp.outer_real_numba(b, c)
    aae(a_plus_b_outer_c_dumb, a_outer_c_plus_b_outer_c_dumb, decimal=10)
    aae(a_plus_b_outer_c_numpy, a_outer_c_plus_b_outer_c_numpy, decimal=10)
    aae(a_plus_b_outer_c_numba, a_outer_c_plus_b_outer_c_numba, decimal=10)


def test_outer_real_scalar_multiplication():
    'verify scalar multiplication property'
    np.random.seed(23)
    a = np.random.rand(3)
    b = np.random.rand(6)
    c = 3.4
    ca_outer_b_dumb = temp.outer_real_dumb(c*a, b)
    a_outer_cb_dumb = temp.outer_real_dumb(a, c*b)
    ca_outer_b_numpy = temp.outer_real_numpy(c*a, b)
    a_outer_cb_numpy = temp.outer_real_numpy(a, c*b)
    ca_outer_b_numba = temp.outer_real_numba(c*a, b)
    a_outer_cb_numba = temp.outer_real_numba(a, c*b)
    aae(ca_outer_b_dumb, a_outer_cb_dumb, decimal=10)
    aae(ca_outer_b_numpy, a_outer_cb_numpy, decimal=10)
    aae(ca_outer_b_numba, a_outer_cb_numba, decimal=10)


def test_outer_real_ignore_complex():
    'complex part of input must be ignored'
    vector_1 = np.ones(5) - 0.4j*np.ones(5)
    vector_2 = np.arange(1,11)
    reference_output = np.resize(vector_2, (vector_1.size, vector_2.size))
    computed_output_dumb = temp.outer_real_dumb(vector_1, vector_2)
    computed_output_numpy = temp.outer_real_numpy(vector_1, vector_2)
    computed_output_numba = temp.outer_real_numba(vector_1, vector_2)
    aae(reference_output, computed_output_dumb, decimal=10)
    aae(reference_output, computed_output_numpy, decimal=10)
    aae(reference_output, computed_output_numba, decimal=10)


def test_outer_complex_invalid_function():
    'fail due to invalid function'
    vector_1 = np.ones(3)
    vector_2 = np.arange(4)
    with pytest.raises(ValueError):
        temp.outer_complex(vector_1, vector_2, function='not_valid_function')


def test_outer_complex_compare_numpy_outer():
    'compare hadamard_complex function with * operator'
    # for matrices
    np.random.seed(21)
    input1 = np.random.rand(7) + 1j*np.random.rand(7)
    input2 = np.random.rand(7) + 1j*np.random.rand(7)
    output_dumb = temp.outer_complex(input1, input2, function='dumb')
    output_numpy = temp.outer_complex(input1, input2, function='numpy')
    output_numba = temp.outer_complex(input1, input2, function='numba')
    output_numpy_outer = np.outer(input1, input2)
    aae(output_dumb, output_numpy_outer, decimal=10)
    aae(output_numpy, output_numpy_outer, decimal=10)
    aae(output_numba, output_numpy_outer, decimal=10)


### matrix-vector product

def test_matvec_real_input_doesnt_match():
    'fail when matrix columns doesnt match vector size'
    A = np.ones((5,4))
    x = np.ones(3)
    with pytest.raises(AssertionError):
        temp.matvec_real_dumb(A, x)
    with pytest.raises(AssertionError):
        temp.matvec_real_numba(A, x)
    with pytest.raises(AssertionError):
        temp.matvec_real_dot(A, x)
    with pytest.raises(AssertionError):
        temp.matvec_real_columns(A, x)


def test_matvec_real_functions_compare_numpy_dot():
    'compare matvec_real_XXXX with numpy.dot'
    np.random.seed(24)
    matrix = np.random.rand(3,4)
    vector = np.random.rand(4)
    output_dumb = temp.matvec_real_dumb(matrix, vector)
    output_numba = temp.matvec_real_numba(matrix, vector)
    output_dot = temp.matvec_real_dot(matrix, vector)
    output_columns = temp.matvec_real_columns(matrix, vector)
    output_numpy_dot = np.dot(matrix, vector)
    aae(output_dumb, output_numpy_dot, decimal=10)
    aae(output_numba, output_numpy_dot, decimal=10)
    aae(output_dot, output_numpy_dot, decimal=10)
    aae(output_columns, output_numpy_dot, decimal=10)


def test_matvec_real_functions_ignore_complex():
    'complex part of input must be ignored'
    np.random.seed(24)
    matrix = np.random.rand(3,4) - 0.3j*np.ones((3,4))
    vector = np.random.rand(4) + 2j*np.ones(4)
    output_dumb = temp.matvec_real_dumb(matrix, vector)
    output_numba = temp.matvec_real_numba(matrix, vector)
    output_dot = temp.matvec_real_dot(matrix, vector)
    output_columns = temp.matvec_real_columns(matrix, vector)
    output_reference = np.dot(matrix.real, vector.real)
    aae(output_dumb, output_reference, decimal=10)
    aae(output_numba, output_reference, decimal=10)
    aae(output_dot, output_reference, decimal=10)
    aae(output_columns, output_reference, decimal=10)


def test_matvec_complex_compare_numpy_dot():
    'compare matvec_complex with numpy.dot'
    np.random.seed(98)
    matrix = np.random.rand(3,4) + 1j*np.random.rand(3,4)
    vector = np.random.rand(4) + 1j*np.random.rand(4)
    output_dumb = temp.matvec_complex(matrix, vector, function='dumb')
    output_numba = temp.matvec_complex(matrix, vector, function='numba')
    output_dot = temp.matvec_complex(matrix, vector, function='dot')
    output_columns = temp.matvec_complex(matrix, vector, function='columns')
    output_numpy_dot = np.dot(matrix, vector)
    aae(output_dumb, output_numpy_dot, decimal=10)
    aae(output_numba, output_numpy_dot, decimal=10)
    aae(output_dot, output_numpy_dot, decimal=10)
    aae(output_columns, output_numpy_dot, decimal=10)


### matrix-matrix product

def test_matmat_real_input_doesnt_match():
    'fail when matrices dont match to compute the product'
    A = np.ones((3,3))
    B = np.ones((4,5))
    with pytest.raises(AssertionError):
        temp.matmat_real_dumb(A, B, check_input=True)
    with pytest.raises(AssertionError):
        temp.matmat_real_numba(A, B, check_input=True)
    with pytest.raises(AssertionError):
        temp.matmat_real_dot(A, B, check_input=True)
    with pytest.raises(AssertionError):
        temp.matmat_real_columns(A, B, check_input=True)
    with pytest.raises(AssertionError):
        temp.matmat_real_outer(A, B, check_input=True)
    with pytest.raises(AssertionError):
        temp.matmat_real_matvec(A, B, check_input=True)


def test_matmat_real_functions_compare_numpy_dot():
    'compare matmat_real_XXXX with numpy.dot'
    np.random.seed(35)
    matrix_1 = np.random.rand(5,3)
    matrix_2 = np.random.rand(3,3)
    output_dumb = temp.matmat_real_dumb(matrix_1, matrix_2)
    output_numba = temp.matmat_real_numba(matrix_1, matrix_2)
    output_dot = temp.matmat_real_dot(matrix_1, matrix_2)
    output_columns = temp.matmat_real_columns(matrix_1, matrix_2)
    output_matvec = temp.matmat_real_matvec(matrix_1, matrix_2)
    output_outer = temp.matmat_real_outer(matrix_1, matrix_2)
    reference = np.dot(matrix_1, matrix_2)
    aae(output_dumb, reference, decimal=10)
    aae(output_numba, reference, decimal=10)
    aae(output_dot, reference, decimal=10)
    aae(output_columns, reference, decimal=10)
    aae(output_matvec, reference, decimal=10)
    aae(output_outer, reference, decimal=10)


def test_matmat_real_functions_ignore_complex():
    'complex part of input must be ignored'
    np.random.seed(35)
    matrix_1 = np.random.rand(5,3)
    matrix_2 = np.random.rand(3,3) - 0.7j*np.ones((3,3))
    output_dumb = temp.matmat_real_dumb(matrix_1, matrix_2)
    output_numba = temp.matmat_real_numba(matrix_1, matrix_2)
    output_dot = temp.matmat_real_dot(matrix_1, matrix_2)
    output_columns = temp.matmat_real_columns(matrix_1, matrix_2)
    output_matvec = temp.matmat_real_matvec(matrix_1, matrix_2)
    output_outer = temp.matmat_real_outer(matrix_1, matrix_2)
    reference = np.dot(matrix_1.real, matrix_2.real)
    aae(output_dumb, reference, decimal=10)
    aae(output_numba, reference, decimal=10)
    aae(output_dot, reference, decimal=10)
    aae(output_columns, reference, decimal=10)
    aae(output_matvec, reference, decimal=10)
    aae(output_outer, reference, decimal=10)


def test_matmat_complex_compare_numpy_dot():
    'compare matmat_complex with numpy.dot'
    np.random.seed(13)
    matrix_1 = np.random.rand(5,3) + 1j*np.random.rand(5,3)
    matrix_2 = np.random.rand(3,3) + 1j*np.random.rand(3,3)
    output_dumb = temp.matmat_complex(matrix_1, matrix_2, function='dumb')
    output_numba = temp.matmat_complex(matrix_1, matrix_2, function='numba')
    output_dot = temp.matmat_complex(matrix_1, matrix_2, function='dot')
    output_columns = temp.matmat_complex(matrix_1, matrix_2, function='columns')
    output_matvec = temp.matmat_complex(matrix_1, matrix_2, function='matvec')
    output_outer = temp.matmat_complex(matrix_1, matrix_2, function='outer')
    reference = np.dot(matrix_1, matrix_2)
    aae(output_dumb, reference, decimal=10)
    aae(output_numba, reference, decimal=10)
    aae(output_dot, reference, decimal=10)
    aae(output_columns, reference, decimal=10)
    aae(output_matvec, reference, decimal=10)
    aae(output_outer, reference, decimal=10)
    
#41##############################################################################################################################
##################################### A partir daqui, testes pedidos pelos exercícios #########################################
###############################################################################################################################

#################################################### test triangle matrix #####################################################

# matvec_triu_prod3 and matvec_triu_prod5 with np.dot

def test_triu_real_functions_compare_numpy_dot():
    'Compare output tril_system (algoritms 3 - 5) and compare tril_system with numpy.dot (reference)' 
    np.random.seed(35)
    matrix = np.random.rand(3,3)
    vector = np.random.rand(3)
    U = np.triu(matrix)
    
    output_triu3 = temp.matvec_triu_prod3(matrix, vector)
    output_triu5 = temp.matvec_triu_prod5(matrix, vector)
    reference = np.dot(U, vector)
    
    # compara o algoritmo 3 com o 5:
    aae(output_triu3, output_triu5, decimal=10)
    
    # compara as saídas dos algoritmos 3 e 5 com a referencia:
    aae(output_triu3, reference, decimal=10)
    aae(output_triu5, reference, decimal=10)

# matvec_tril_prod8 and matvec_tril_prod10 with np.dot

def test_tril_real_functions_compare_numpy_dot():
    'Compare output tril_system (algoritms 8 - 10) and compare tril_system with numpy.dot (reference)'
    np.random.seed(35)
    matrix = np.random.rand(3,3)
    vector = np.random.rand(3)
    L = np.tril(matrix)
    
    # compara o algoritmo 8 com o 10:
    output_tril8 = temp.matvec_tril_prod8(matrix, vector)
    output_tril10 = temp.matvec_tril_prod10(matrix, vector)
    reference = np.dot(L, vector)
    
    # compara as saídas dos algoritmos 8 e 10 com a referencia:
    aae(output_tril8, output_tril10, decimal=10)
    
    aae(output_tril8, reference, decimal=10)
    aae(output_tril10, reference, decimal=10)
    
############################################## test triangle system ################################################

def test_triu_system_1():
    'compare matmat_real_XXXX with numpy.dot'
    np.random.seed(35)
    matrix = np.random.rand(3,3)
    vector_x = np.random.rand(3)
    U = np.triu(matrix)
    
    output_triup = temp.matvec_triu_prod3(matrix, vector_x) # gerando um y de y = A.x
    output_triu_sys = temp.triu_system(matrix, output_triup) # gerando o x 
    
    aae(output_triu_sys, vector_x, decimal=10) # comparando o x e vector_x (original)

def test_triu_system_2():
    'compare matmat_real_XXXX with numpy.dot'
    np.random.seed(35)
    matrix = np.random.rand(3,3)
    vector_x = np.random.rand(3)
    U = np.triu(matrix)
    
    output_triup = temp.matvec_triu_prod3(matrix, vector_x) # gerando um y de y = A.x
    output_triu_sys = temp.triu_system(matrix, output_triup) # gerando o x 
    reference = np.linalg.solve(U, output_triup) # gerando o x0
    
    aae(output_triu_sys, reference, decimal=10) # comparando o x e o x0 (referencia)
    
def test_tril_system_1():
    'compare matmat_real_XXXX with numpy.dot'
    np.random.seed(35)
    matrix = np.random.rand(3,3)
    vector_x = np.random.rand(3)
    L = np.tril(matrix)
    
    output_trilow = temp.matvec_tril_prod8(matrix, vector_x)
    output_tril_sys = temp.tril_system(matrix, output_trilow) # gerando o x
    
    aae(output_tril_sys, vector_x, decimal=10) # comparando o x com vector_x (original)
    
def test_tril_system_2():
    'compare matmat_real_XXXX with numpy.dot'
    np.random.seed(35)
    matrix = np.random.rand(3,3)
    vector_x = np.random.rand(3)
    L = np.tril(matrix)
    
    output_trilow = temp.matvec_tril_prod8(matrix, vector_x)
    output_tril_sys = temp.tril_system(matrix, output_trilow) # gerando o x
    reference = np.linalg.solve(L, output_trilow) # gerando o x0
    
    aae(output_tril_sys, reference, decimal=10) # comparando o x e o x0 (referencia)
    
    
#################################################### test Gauss_elim #####################################################

def test_gauss_elim_dumb_1():
    'Compare Gauss_elim_dumb (EXTRA function) with equivalent triangular system'
    np.random.seed(35)
    matrix = np.array([[-3.,-1.,2.], # obs: matrix específica que não precisa de permutação!
                       [-2.,1.,2.],
                       [2.,1.,-1.]]) # A0
    
    vector_y = np.random.rand(3) # y0
    
    I, L_ref, U_ref = lu(matrix)
    U_cal, y_cal = temp.Gauss_elim_dumb(matrix, vector_y) 
    
    aae(U_cal, U_ref, decimal=10) # comparando o U_cal e o U_ref
    
def test_gauss_elim_dumb_2():
    'Compare Gauss_elim_dumb (EXTRA function) with solution linalg.solve.'
    np.random.seed(35)
    matrix = np.array([[-3.,-1.,2.], # obs: matrix específica que não precisa de permutação!
                       [-2.,1.,2.],
                       [2.,1.,-1.]]) # A0
    
    vector_y = np.random.rand(3) # y0
    
    U_cal, y_cal = temp.Gauss_elim_dumb(matrix, vector_y) 
    x_cal = temp.triu_system(U_cal, y_cal)
    x_ref = np.linalg.solve(matrix, vector_y)
    
    aae(x_cal, x_ref, decimal=10) # comparando o U_cal e o U_ref
    
def test_gauss_elim_1():
    'Compare Gauss_elim with equivalent triangular system'
    np.random.seed(35)
    matrix = np.random.rand(5,5) # A0
    vector_y = np.random.rand(5) # y0
    
    I, L_ref, U_ref = lu(matrix)
    
    C, y = temp.Gauss_elim(matrix, vector_y) 
    U_cal = np.triu(C)
    
    aae(U_cal, U_ref, decimal=10) # comparando o U_cal e o U_ref
    
def test_gauss_elim_2():
    'Compare Gauss_elim with numpy function... And compare x1 (calculated) with x (initial)'
    np.random.seed(35)
    matrix = np.random.rand(5,5) # A0
    vector_x = np.random.rand(5) # x0
    
    vector_y = np.dot(matrix, vector_x) #y0
    
    Gauss_A, Gauss_y = temp.Gauss_elim(matrix, vector_y) # gerando a matrix equivalente (met. Gauss)
    x1 = temp.triu_system(Gauss_A, Gauss_y) # gerando um x1 através da função de solução de sistemas do template
    x = np.linalg.solve(matrix, vector_y) # geradno um x através de uma função do numpy
    
    aae(x1, vector_x, decimal=10) # comparando o x1 e o x0
    aae(x, x1, decimal=10) # comparando o x e o x0
    

def test_gauss_elim_3():
    'test to expected value of function "permut"'
    A_inp = np.array([[2.,1.,-1.],
                      [-3.,-1.,2.],
                      [-2.,1.,2.]])
    
    A_ref = np.array([[-3., -1.,  2.],
                      [2.,  1., -1.],
                      [-2.,  1.,  2.]])
    
    N = A_inp.shape[0]
    for k in range(N):
        p, A_inp = temp.permut(A_inp,k)
    
    aae(A_ref, A_inp, decimal=10)
    
    
################################################## test Gauss_expanded ###################################################

def test_gauss_elim_expanded_1():
    'Two testes for Gauss_elim_expanded:'
    '1°) compare Identiry cal with Identity reference'
    
    np.random.seed(35)
    A = np.random.rand(5,5) # matrix A
    I = np.identity(5) # identidade
    
    B, Z = temp.Gauss_elim_expanded(A, I) # gerando a matrix expandida C = B,Z
    A_inv = temp.Gauss_solution_inv(B, Z) # matriz inversa usando B e Z usando a função Gauss_solution_inv (stacking)
    I1 = np.dot(A,A_inv) # Identidade método 1
    I2 = np.dot(A_inv,A) # Identidade método 2
    
    aae(I1,I, decimal=5) # comparando o I com I1 calculado 
    aae(I2,I, decimal=5) # comparando o I com I2 calculado
    
def test_gauss_elim_expanded_2():
    'Two testes for Gauss_elim_expanded:'
    '2°) Compare A inverse calculated with a reference'
    
    np.random.seed(35)
    A = np.random.rand(5,5) # matrix A
    I = np.identity(5) # identidade
    
    B, Z = temp.Gauss_elim_expanded(A, I) # gerando a matrix expandida C = B,Z
    A_inv = temp.Gauss_solution_inv(B, Z) # matriz inversa usando B e Z usando a função Gauss_solution_inv (stacking)
    A_ref = np.linalg.inv(A) # gerando a A_inv com a função do numpy
    
    aae(A_inv, A_ref, decimal=5) # comparando o resultado da Inversão calculado com o do numpy 
    
    
######################################################### test LU ########################################################

def test_lu_decomp_test_1():
    'Compare L*U with original A matrix'
        
    np.random.seed(35)
    A = np.random.rand(5,5) # matrix A
    y = np.random.rand(5) # vetor y
    
    D = temp.lu_decomp(A)
    L, U, x = temp.lu_solve(D, y)
    A_cal = np.dot(L,U)
        
    aae(A_cal, A, decimal=10) # comparando o resultado da Inversão calculado com o do numpy
    

def test_lu_decomp_test_2():
    'Compare de solution LU with reference linalg.solve'
        
    np.random.seed(35)
    A = np.random.rand(5,5) # matrix A
    y = np.random.rand(5) # vetor y
    
    D = temp.lu_decomp(A)
    L, U, x1 = temp.lu_solve(D, y)
    x2 = np.linalg.solve(A, y)
    
    aae(x1, x2, decimal=10) # comparando as saídas x1 e x2
    
    
def test_lu_decomp_test_3():
    'Create A0x0 = y0 and compare with x1 (result - lu_decomp and lu_solve functions)'
        
    np.random.seed(35)
    A0 = np.random.rand(5,5) # matrix A
    x0 = np.random.rand(5) # vetor x
    
    y0 = np.dot(A0,x0) # gerar y
    
    D = temp.lu_decomp(A0)
    L, U, x1 = temp.lu_solve(D, y0)
    
    aae(x0, x1, decimal=10) # comparando as saídas x0 e x1
    
    
def test_lu_decomp_alt_extra():
    'Test to compare L and U arrays of extra lu_decomp_alt function with Python "lu" function.'
        
    np.random.seed(35)
    matrix = np.array([[-3.,-1.,2.], # obs: matrix específica que não precisa de permutação!
                       [-2.,1.,2.],
                       [2.,1.,-1.]]) # A0
    
    vector_y = np.random.rand(3) # y0
    
    L1,U1 = temp.lu_decomp_alt(matrix)
    I, L2, U2 = lu(matrix)
        
    aae(L1, L2, decimal=10) # comparando o resultado da matrix L do template com o no python
    aae(U1, U2, decimal=10) # comparando o resultado da matrix U do template com o no python
    
    
################################################### test LU com pivoteamento ##################################################

def test_lu_decomp_pivoting_1():
    'Compare the result PA and LU'
        
    np.random.seed(35)
    A = np.random.rand(5,5) # matrix A
    y = np.random.rand(5) # vetor x
    
    P, C = temp.lu_decomp_pivoting(A)
    L, U, x = temp.lu_solve_pivoting(P, C, y)
    
    PA = np.dot(P, A)
    LU = np.dot(L, U)
    
    aae(PA, LU, decimal=10) # comparando as saídas PA e LU
    
def test_lu_decomp_pivoting_2():
    'Compare the output of lu_decomp_pivoting and scipy.linalg.lu.'
        
    np.random.seed(35)
    A = np.random.rand(5,5) # matrix A
    y = np.random.rand(5) # vetor y
    
    P, C = temp.lu_decomp_pivoting(A)
    L, U, x = temp.lu_solve_pivoting(P, C, y)
    x_ref = np.linalg.solve(A, y)
    
    PP, LL, UU = lu(A)
    
    
    aae(P, PP.T, decimal=10) # comparando as saídas PA e LU
    aae(L, LL, decimal=10) # comparando as saídas PA e LU
    aae(U, UU, decimal=10) # comparando as saídas PA e LU
    aae(x, x_ref, decimal=10) # comparando as saídas PA e LU
    

def test_lu_decomp_pivoting_3():
    'Cal for A0x0 = y0 -> using lu_decomp_pivoting and lu_solve_pivoting and ->x1. Compare x0 with x1'
        
    np.random.seed(35)
    A0 = np.random.rand(5,5) # matrix A
    x0 = np.random.rand(5) # vetor x
    
    y0 = np.dot(A0, x0)

    P, C = temp.lu_decomp_pivoting(A0)
    L, U, x1 = temp.lu_solve_pivoting(P, C, y0)
    
    aae(x0, x1, decimal=10) # comparando as saídas x1 e x0

    

########################################################### test LDLt ############################################################

def test_ldlt_decomp():
    'Compare A with LDLt'
        
    np.random.seed(35)
    A0 = np.random.rand(3,3) + np.identity(3) 
    A0 = 0.5*(A0 + A0.T)
    y0 = np.random.rand(3) # vetor y
    
    L, d = temp.ldlt_decomp(A0) # L e d da função ldlt
    D = np.diag(d) # gerando a matrix D a partir de d
    
    # calculando LDLt usando a multiplicação de matrix 
    DLt = temp.matmat_real_dot(D, L.T)
    LDLt = temp.matmat_real_dot(L, DLt) 
    
    aae(A0, LDLt, decimal=10) # comparando as saídas A0 e LDLt
    
    
def test_ldlt_solve():
    'Compare ldlt with np.linalg.solve'
        
    np.random.seed(35)
    A0 = np.random.rand(3,3) + np.identity(3) 
    A0 = 0.5*(A0 + A0.T)
    
    y0 = np.random.rand(3) # vetor y
    
    L, d = temp.ldlt_decomp(A0) # L e d da função ldlt
    x = temp.ldlt_solve(L,d,y0)
    x_ref = np.linalg.solve(A0, y0)
    
    aae(x, x_ref, decimal=10) # comparando as saídas A0 e LDLt
    
    
def test_ldlt_inv():
    'Compare AA^-1 = identity using idlt_inverse'
        
    np.random.seed(35)
    A0 = np.random.rand(3,3) + np.identity(3) 
    A0 = 0.5*(A0 + A0.T)
    I0 = np.identity((3))
    
    A_inv = temp.ldlt_inverse(A0) # inversa de A0 com a função criada
    I_cal = np.dot(A0, A_inv)
    A_inv_ref = np.linalg.inv(A0) # inversa com a função do numpy
    
    aae(A_inv, A_inv_ref, decimal=10) # comparando as saídas das inversas
    aae(I0, I_cal, decimal=10) # comparação entre as identidades calculadas e de referências
    
    
########################################################### test Cholesky ############################################################

def test_cho_decomp():
    'Compare A with GG.T'
        
    np.random.seed(35)
    A0 = np.random.rand(3,3) + np.identity(3) 
    A0 = 0.5*(A0 + A0.T)
    y0 = np.random.rand(3) # vetor y
    
    G = temp.cho_decomp(A0) # G da função Cho
    
    # calculando GGt usando a muitiplicação de matrix 
    GGt = temp.matmat_real_dumb(G, G.T)
    
    aae(A0, GGt, decimal=10) # comparando as saídas A0 e LDLt
    

def test_cho_solve():
    'Compare cho_solve with np.linalg.solve'
        
    np.random.seed(35)
    A0 = np.random.rand(3,3) + np.identity(3) 
    A0 = 0.5*(A0 + A0.T)
    y0 = np.random.rand(3) # vetor y
    
    G = temp.cho_decomp(A0) # G da função Cho
    x = temp.cho_solve(G, y0) # solução x usando cholesky
    x_ref = np.linalg.solve(A0, y0)
    
    aae(x, x_ref, decimal=10) # comparando as saídas A0 e LDLt
    
    
def test_Cho_inv():
    'Compare AA^-1 = identity using Cho_inverse'
        
    np.random.seed(35)
    A0 = np.random.rand(3,3) + np.identity(3) 
    A0 = 0.5*(A0 + A0.T)
    I0 = np.identity((3))
    
    A_inv = temp.cho_inverse(A0) # inversa de A0 com a função criada
    A_inv_ref = np.linalg.inv(A0) # inversa com a função do numpy
    
    aae(A_inv, A_inv_ref, decimal=10) # comparando as saídas das inversas
    

###################################################### test Least Squares ########################################################

def test_straight_line_matrix():
    'compare the matrix A generated by the function "straight_line_matrix" with A for numpy'
        
    np.random.seed(35)
    x = np.random.rand(3) 
    
    A_ref = np.polynomial.polynomial.polyvander(x, deg=1)# função do numpy para criar a matrix A
    A = temp.straight_line_matrix(x) # função no template pra gerar a matrix A
    
    aae(A_ref, A, decimal=10) 
    
def test_straight_line():
    'Compare input parameters with estimated parameters'
        
    np.random.seed(35)
    x = np.random.rand(3)
    a = 2.5
    b = 3.
    p_true = np.array([b, a])
    d = a*x + b
    
    p_estimated = temp.straight_line(x, d) # função no template pra gerar a matrix A
    
    aae(p_true, p_estimated, decimal=10)
    
def test_parameter_covariance():
    'test to compare the function "parameter_covariance" (W_cal) with covariance using numpy (W_ref)'
    
    x = np.linspace(1., 10., 5) # gerando os dados
    A_ref = np.array([[1., 1.],[1., 3.25],[1., 5.5],[1., 7.75],[1., 10.]]) # matrix A com os dados x
    sig = 0.2 #definindo sigma
    
    # Calculando covariância usando funções do numpy:
    ATA = np.dot(A_ref.T, A_ref)
    ATA_inv = np.linalg.inv(ATA) 
    W_ref = (sig**2)*ATA_inv
    
    # Calculando covariância usando a função do template:
    W_cal = temp.parameter_covariance(sig, A_ref)
    
    aae(W_ref, W_cal, decimal=10) # comparando as saídas
    
    
###################################################### test Fourier Series ########################################################

def test_DFT_Matrix_1():
    'Compare the result produced by DFT_matrix and an expected result produced by a specific input'
    
    H_unscaled_ref = np.array([[ 1. +0.j       ,  1. +0.j       ,  1. +0.j       ],
                               [ 1. +0.j       , -0.5-0.8660254j, -0.5+0.8660254j],
                               [ 1. +0.j       , -0.5+0.8660254j, -0.5-0.8660254j]])
    
    H_n_ref = np.array([[ 0.33333333+0.j        ,  0.33333333+0.j        ,0.33333333+0.j        ],
                        [ 0.33333333+0.j        , -0.16666667-0.28867513j,-0.16666667+0.28867513j],
                        [ 0.33333333+0.j        , -0.16666667+0.28867513j,-0.16666667-0.28867513j]])
    
    H_sqrtn_ref = np.array([[ 0.57735027+0.j ,  0.57735027+0.j ,  0.57735027+0.j ],
                            [ 0.57735027+0.j , -0.28867513-0.5j, -0.28867513+0.5j],
                            [ 0.57735027+0.j , -0.28867513+0.5j, -0.28867513-0.5j]])
    
    
    N = 3
    H1_cal = temp.DFT_matrix(N, scale=None)
    H2_cal = temp.DFT_matrix(N, scale='n')
    H3_cal = temp.DFT_matrix(N, scale='sqrtn')
    
    aae(H1_cal, H_unscaled_ref, decimal=5)
    aae(H2_cal, H_n_ref, decimal=5)
    aae(H3_cal, H_sqrtn_ref, decimal=5)
    
def test_DFT_Matrix_2():
    'Compare the result produced by DFT_Matrix and the result produced by the routine scipy.linalg.dft'
    
    np.random.seed(35)
    N = 5
    g = np.random.rand(N) + 1j*np.random.rand(N)
    
    H_1 = temp.DFT_matrix(N, scale=None, conjugate=False)
    H_2 = temp.DFT_matrix(N, scale='sqrtn', conjugate=False)
    H_3 = temp.DFT_matrix(N, scale='n', conjugate=False)
    
    FN_unscaled = dft(N, scale=None)
    FN_sqrtn = dft(N, scale='sqrtn')
    FN_n = dft(N, scale='n')
    
    aae(H_1, FN_unscaled, decimal=10)
    aae(H_2, FN_sqrtn, decimal=10)
    aae(H_3, FN_n, decimal=10)
    
    
def test_DFT_Matrix_3():
    'Satisfy the following conditions: F^∗N = FHN, F^−1N = 1/N.F^∗N and NI = F^HN.FN.'
    
    np.random.seed(35)
    N = 5
    g = np.random.rand(N) + 1j*np.random.rand(N)
    
    H_conj_true = temp.DFT_matrix(N, scale=None, conjugate=True)
    H_conj_false = temp.DFT_matrix(N, scale=None, conjugate=False)
    H_inv = np.linalg.inv(H_conj_false)
    NI = N*np.identity(N)
    HH = np.dot(H_conj_true.T, H_conj_false)
    
    aae(H_conj_true, H_conj_true.T, decimal=10)
    aae(H_inv, (1/N)*H_conj_true, decimal=10)
    aae(NI, HH, decimal=10)
    
    
def test_fft1D_1():
    'Compare the result produced by fft1D and the result produced by the routine scipy.fft.fft'
    
    np.random.seed(35)
    N = 4
    g = np.random.rand(N) + 1j*np.random.rand(N)
    
    FFT_1 = temp.fft1D(g, scale= None, conjugate=False)
    FFT_2 = temp.fft1D(g, scale='n', conjugate=False)
    FFT_3 = temp.fft1D(g, scale='sqrtn', conjugate=False)
    
    G_unscaled_sp = fft(g, norm=None)
    G_sqrtn_sp = fft(g, norm='ortho')
    
    aae(FFT_1, G_unscaled_sp, decimal=5)
    aae(FFT_3, G_sqrtn_sp, decimal=5)
    
    
def test_ifft1D_1():
    'Compare the result produced by ifft1D and the result produced by the routine scipy.fft.ifft'
    
    np.random.seed(35)
    N = 4
    g = np.random.rand(N) + 1j*np.random.rand(N)
    
    FFT_1 = temp.ifft1D(g, scale=None, conjugate=False)
    FFT_2 = temp.ifft1D(g, scale='n', conjugate=False)
    FFT_3 = temp.ifft1D(g, scale='sqrtn', conjugate=False)
    
    G_None = ifft(g, N, norm=None)
    G_sqrtn = ifft(g, N, norm='ortho')
    
    aae(FFT_1, G_None, decimal=5)
    aae(FFT_3, G_sqrtn, decimal=5)
    
def test_ifft1D_2():
    'Create a real data vector g, compute its DFT G with the function fft1D and then compare the computed the IDFT of G with the original  vector g by using your function ifft1D.'
    
    np.random.seed(35)
    N = 4
    g = np.random.rand(N) + 1j*np.random.rand(N)
    
    G1 = temp.fft1D(g, scale=None,  conjugate=False)
    G2 = temp.fft1D(g, scale='n',  conjugate=False)
    G3 = temp.fft1D(g, scale='sqrtn',  conjugate=False)
    
    gg1 = temp.ifft1D(G1, scale=None, conjugate=False)
    gg2 = temp.ifft1D(G2, scale='n', conjugate=False)
    gg3 = temp.ifft1D(G3, scale='sqrtn', conjugate=False)
    
    aae(gg1, g, decimal=5)
    aae(gg2, g, decimal=5)
    aae(gg3, g, decimal=5)
    
    
    
###################################################### test Convolution ########################################################
    
def conv_circular_dft_compare():
    'comparing the results produced by functions circular_convolution_matvec and circular_convolution_dft.'
    a = np.random.rand(4)
    b = np.random.rand(4)
    M = np.size(a)
    
    w1 = temp.circular_convolution_dft(a,b)
    w2 = temp.circular_convolution_matvec(a,b, function='dot').real
    
    aae(w1, w2, decimal=10)
    
    
def conv_linear_dft_compare_1():
    'comparing the results produced by functions linear_convolution_matvec and linear_convolution_dft.'
    a = np.random.rand(4)
    b = np.random.rand(4)
    M = np.size(a)
    
    w1 = temp.linear_convolution_dft(a,b)
    w2 = temp.linear_convolution_matvec(a,b, function='dot').real
    
    aae(w1, w2, decimal=10)
    
    
def conv_linear_dft_compare_2():
    'comparing the results produced by functions linear_convolution_matvec and linear_convolution_dft.'
    a = np.random.rand(4)
    b = np.random.rand(4)
    M = np.size(a)
    
    w1 = temp.linear_convolution_dft(a,b)
    w2 = temp.linear_convolution_matvec(a,b, function='dot').real
    w_ref = np.convolve(a,b)
    
    aae(w1, w_ref, decimal=10)
    aae(w2, w_ref,decimal=10)
    
    
def conv_linear_and_circular():
    'comparing the results produced by your functions circular_convolution_dft and linear_convolution_dft'
    
    a = np.random.rand(4)
    b = np.random.rand(4)
    
    w1 = temp.circular_convolution_dft(a,b)
    w2 = temp.linear_convolution_dft(a,b)
    
    aae(w1, w2, decimal=10)

    
def test_1_correlation_():
    'compare correlation_dft with a matrix-vector computed crosscorrelation'
    np.random.seed(36)
    a = np.random.rand(5)
    b = np.random.rand(5)
    
    output_dft = temp.correlation_dft(a, b)
    output_matvec =temp.correlation_matvec(a, b)
    
    aae(output_dft, output_matvec, decimal=10)

    
def test_2_correlation():
    'compare correlation_dft with numpy correlate'
    np.random.seed(37)
    a = np.random.rand(5)
    b = np.random.rand(5)
    
    output_dft = temp.correlation_dft(a, b)
    reference_output_numpy = np.correlate(a, b, mode='full')
    
    aae(output_dft, reference_output_numpy, decimal=10)
    
