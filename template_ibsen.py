###################################################### Ibsen P. S. Gomes ############################################################
########################################## Mátodos Numéricos - Obseratório Nacional #################################################
#################################################### Template de funções ############################################################

# Bibliotecas:
import numpy as np
from numba import njit
from scipy.linalg import dft
from scipy.linalg import toeplitz, circulant, dft


##################################################### SMA function #################################################################

def sma1d(data, window_size):
    '''
    Apply a simple moving average filter with
    size window_size to data.
    input
    data: numpy array 1D - data set to be filtered.
    window_size: int - number of points forming the window.
                 It must be odd and greater than 3.
    output
    filtered_data: numpy array 1D - filtered data. This array has the
                   same number of elementos of the original data.
    '''

    assert data.size >= window_size, \
        'data must have more elements than window_size'

    assert window_size%2 != 0, 'window_size must be odd'

    assert window_size >= 3, 'window_size must be greater than or equal to 3'

    # lost points at the extremities
    i0 = window_size//2

    # non-null data
    N = data.size - 2*i0

    filtered_data = np.empty_like(data)

    filtered_data[:i0] = 0.
    filtered_data[-1:-i0-1:-1] = 0.

    for i in range(N):
        filtered_data[i0+i] = np.mean(data[i:i+window_size])

    return filtered_data


###################################################### Scalar-vector product ####################################################

# Cálculo y = a*x real de forma simples:

def scalar_vec_real_dumb(a, x, check_input=True):
    '''
    Compute the product of a scalar a and vector x, where
    a is real and x is in R^N. The imaginary parts are ignored.
    The code uses a simple "for" to iterate on the array.
    Parameters
    ----------
    a : scalar
        Real number.
    x : array 1D
        Vector with N elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array
        Product of a and x.y[i] = a*x.real[i]
    '''
    
    'Teste de entrada:'
    a = np.asarray(a) # asarray ja converte para a análise de assert a seguir
    x = np.asarray(x)
    if check_input is True:
        assert a.ndim == 0, 'a must be a scalar' # dim 0 = constante
        assert x.ndim == 1, 'x must be a 1D' # dim 1 = vetor e dim 2 = matriz
        
    'Laço para operação linear:'
    #y = np.zeros_like(x) # criação de um vetor nulo com tamanho de "x"
    result = np.empty_like(x)
    for i in range(x.size):
        # the '.real' forces the code to use
        # only the real part of the arrays
        result[i] = a.real*x.real[i]

    return result


# Calculo y = a*x real utilizando numpy:

def scalar_vec_real_numpy(a, x, check_input=True):
    '''
    Compute the product of a scalar a and vector x, where
    a is real and x is in R^N. The imaginary parts are ignored.
    The code uses numpy.
    Parameters
    ----------
    a : scalar
        Real number.
    x : array 1D
        Vector with N elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array
        Product of a and x.
    '''
    a = np.asarray(a)
    x = np.asarray(x)
    if check_input is True:
        assert a.ndim == 0, 'a must be a scalar'
        assert x.ndim == 1, 'x must be a 1D'

    result = a.real*x.real

    return result


# Calculo y = a*x real utilizando numba:
@njit
def scalar_vec_real_numba(a, x, check_input=True):
    '''
    Compute the product of a scalar a and vector x, where
    a is real and x is in R^N. The imaginary parts are ignored.
    The code uses numba.
    Parameters
    ----------
    a : scalar
        Real number.
    x : array 1D
        Vector with N elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array
        Product of a and x.
    '''
    a = np.asarray(a)
    x = np.asarray(x)
    if check_input is True:
        assert a.ndim == 0, 'a must be a scalar'
        assert x.ndim == 1, 'x must be a 1D'

    result = np.empty_like(x)
    for i in range(x.size):
        # the '.real' forces the code to use
        # only the real part of the arrays
        result[i] = a.real*x.real[i]

    return result

# Calculo com variáveis complexas:

def scalar_vec_complex(a, x, check_input=True, function='dumb'):
    '''
    Compute the dot product of a is a complex number and x
    is a complex vector.
    Parameters
    ----------
    a : scalar
        Complex number.
    x : array 1D
        Complex vector with N elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real scalar-vector product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'dumb'.
    Returns
    -------
    result : scalar
        Product of a and x.
    '''
    a = np.asarray(a)
    x = np.asarray(x)
    if check_input is True:
        assert a.ndim == 0, 'a must be a scalar'
        assert x.ndim == 1, 'x must be a 1D'

    scalar_vec_real = {
        'dumb': scalar_vec_real_dumb,
        'numpy': scalar_vec_real_numpy,
        'numba': scalar_vec_real_numba
    }
    if function not in scalar_vec_real:
        raise ValueError("Function {} not recognized".format(function))

    result_real = scalar_vec_real[function](a.real, x.real, check_input=False)
    result_real -= scalar_vec_real[function](a.imag, x.imag, check_input=False)
    result_imag = scalar_vec_real[function](a.real, x.imag, check_input=False)
    result_imag += scalar_vec_real[function](a.imag, x.real, check_input=False)

    result = result_real + 1j*result_imag

    return result


################################################################ dot ###########################################################

# dot simples:

def dot_real_dumb(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.
    The code uses a simple "for" to iterate on the arrays.
    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array'
        assert x.shape == y.shape, 'x and y must be a same shape'
        
    s = 0
    for i in range(x.size):
        s += x.real[i]*y.real[i]  
        
    return s


@njit
def dot_real_numba(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.
    The code uses numba jit.
    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
        
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array'
        assert x.shape == y.shape, 'x and y must be a same shape'
        
    s = 0
    for i in range(x.size):
        s += x.real[i]*y.real[i]

    return s


# dot numpy:

def dot_real_numpy(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.
    The code uses numpy.sum.
    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    
    s = 0
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array'
        assert x.shape == y.shape, 'x and y must be a same shape'
         
    s = np.sum(x.real*y.real)
    
    return s


# dot complexo:

def dot_complex_dumb(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of C^N.
    The code uses a simple "for" to iterate on the arrays.
    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array' 
        assert x.shape == y.shape, 'x and y must be a same shape'

        
    result_real = dot_real_dumb(x.real, y.real, check_input=False)
    result_real -= dot_real_dumb(x.imag, y.imag, check_input=False)
    result_imag = dot_real_dumb(x.real, y.imag, check_input=False)
    result_imag += dot_real_dumb(x.imag, y.real, check_input=False)

    result = result_real + 1j*result_imag

    return result


def dot_complex_numpy(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of C^N.
    The code uses numpy.sum.
    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.
    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array' 
        assert x.shape == y.shape, 'x and y must be a same shape'


    result_real = dot_real_numpy(x.real, y.real, check_input=False)
    result_real -= dot_real_numpy(x.imag, y.imag, check_input=False)
    result_imag = dot_real_numpy(x.real, y.imag, check_input=False)
    result_imag += dot_real_numpy(x.imag, y.real, check_input=False)

    result = result_real + 1j*result_imag

    return result  


def dot_complex_numba(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of C^N.
    The code uses numba.
    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.
    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array' 
        assert x.shape == y.shape, 'x and y must be a same shape'


    result_real = dot_real_numba(x.real, y.real, check_input=False)
    result_real -= dot_real_numba(x.imag, y.imag, check_input=False)
    result_imag = dot_real_numba(x.real, y.imag, check_input=False)
    result_imag += dot_real_numba(x.imag, y.real, check_input=False)

    result = result_real + 1j*result_imag

    return result  


def dot_complex(x, y, conjugate=False, check_input=True, function='dumb'):
    '''
    Compute the dot product of a is a complex number and x
    is a complex vector.
    Parameters
    ----------
    x : array 1D
        Complex vector 
    y : array 1D
        Complex vector 
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing dot product.
        The function name must be 'dot-dumb', 'dot-numpy' or 'dot-numba'.
        Default is 'dumb' 
    Returns
    -------
    result : dot
        Product of x and y.
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array' 
        assert x.shape == y.shape, 'x and y must be a same shape'
        
    # dicionário para armazenar as 3 formas de calcular dot:
    dot_vec_real = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'numba': dot_real_numba,
    }
    if function not in dot_vec_real:
        raise ValueError("Function {} not recognized".format(function))

    if conjugate is True:
        result_real = dot_vec_real[function](x.real, y.real, check_input=False)
        result_real += dot_vec_real[function](x.imag, y.imag, check_input=False)
        result_imag = dot_vec_real[function](x.real, y.imag, check_input=False)
        result_imag -= dot_vec_real[function](x.imag, y.real, check_input=False)
    else:
        result_real = dot_vec_real[function](x.real, y.real, check_input=False)
        result_real -= dot_vec_real[function](x.imag, y.imag, check_input=False)
        result_imag = dot_vec_real[function](x.real, y.imag, check_input=False)
        result_imag += dot_vec_real[function](x.imag, y.real, check_input=False)
    result = result_real + 1j*result_imag

    return result


########################################################## Hadamard produt #########################################################

# Hadamard dumb:

def hadamard_real_dumb(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.
    The code uses a simple doubly nested loop to iterate on the arrays.
    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1 or x.ndim == 2, 'x must be a 1D or 2D array'
        assert x.shape == y.shape, 'x and y must be same shape' 
        
    N = np.size(x)
    
    if x.ndim == 1: 
        z = np.zeros(N)
        for i in range(N):
            z[i] = x.real[i]*y.real[i]
            
    if x.ndim == 2: 
        N, M = np.shape(x)
        z = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                z[i,j] = x.real[i,j]*y.real[i,j]
            
    return z


# Hadamard numpy:

def hadamard_real_numpy(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.
    The code uses the asterisk (star) operator.
    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1 or x.ndim == 2, 'x must be a 1D or 2D array'
        assert x.shape == y.shape, 'x and y must be same shape'
        
    N = np.size(x)
    
    if x.ndim == 1:
        z = np.zeros(N)
        z[:] = x.real[:]*y.real[:]
        
    if x.ndim == 2:
        N, M = np.shape(x)
        z = np.zeros((N,M))
        z[:,:] = x.real[:,:]*y.real[:,:]
        
    return z


# Pelo método numba:

@njit
def hadamard_real_numba(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.
    The code uses numba.
    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1 or x.ndim == 2, 'x must be a 1D or 2D array'
        assert x.shape == y.shape, 'x and y must be same shape'
        
    N = np.size(x)
    
    if x.ndim == 1:
        z = np.zeros(N)
        for i in range(N):
            z[i] = x.real[i]*y.real[i]
        
    if x.ndim == 2:
        N, M = np.shape(x)
        z = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                z[i,j] = x.real[i,j]*y.real[i,j]
        
    return z


# Hadamard matrix:

def hadamard_real_matrix(A, B, check_input=True):
    '''
    Hadamard product between two matrixs
    
    imput:
    A = 2D array
    B = 2D array
    
    operation:
    multiplicação de cada termo das matrizes -> A[i] * B[i]
    
    output:
    z = 2D array
    '''
    A = np.asarray(A)
    B = np.asarray(B)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D array'
        assert A.shape == B.shape, 'A and B must be same shape'
        
    M = int(np.sqrt(np.size(A)))
    z = np.zeros((M,M))
    z[:][:] = A.real[:][:]*B.real[:][:]
    
    return z


# Hadamard complexo:

def hadamard_complex(x, y, check_input=True, function='dumb'):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be complex vectors or matrices having the same shape.
    Parameters
    ----------
    x, y : arrays
        Complex vectors or matrices having the same shape.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real Hadamard product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'dumb'.
    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1 or x.ndim == 2, 'x must be a 1D or 2D array'
        assert x.shape == y.shape, 'x and y must be same shape'
        
    hadamard_vec_real = {
        'dumb': hadamard_real_dumb,
        'numpy': hadamard_real_numpy,
        'numba': hadamard_real_numpy
    }
    if function not in hadamard_vec_real:
        raise ValueError("Function {} not recognized".format(function))
        
    result_real = hadamard_vec_real[function](x.real, y.real, check_input=False)
    result_real -= hadamard_vec_real[function](x.imag, y.imag, check_input=False)
    result_imag = hadamard_vec_real[function](x.real, y.imag, check_input=False)
    result_imag += hadamard_vec_real[function](x.imag, y.real, check_input=False)

    result = result_real + 1j*result_imag

    return result


######################################################### Outer products #####################################################

# operação vet x vet = matrix (simples):

def outer_real_dumb(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.
    The code uses a simple "for" to iterate on the arrays.
    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
        
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array'
        assert y.ndim == 1, 'y must be a 1D array'
        
    O = np.size(x) # dimensão do vetor x
    P = np.size(y) # dimensão do vetor y
    Q = np.zeros((O,P)) # matrix de zeros com dimensões (NxM) 
    
    for i in range (O):
        for j in range (P):
            Q[i][j] = x.real[i]*y.real[j]
            
    return Q


# Usando ":" para automatizar a operação em linhas ou colunas (numpy?):

def outer_real_numpy(x, y, check_input=True, function='numpy'):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.
    The code uses numpy.newaxis for broadcasting
    https://numpy.org/devdocs/user/theory.broadcasting.html
    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array'
        assert y.ndim == 1, 'y must be a 1D array'
        
    O = np.size(x) # dimensão do vetor x
    P = np.size(y) # dimensão do vetor y
    Q = np.zeros((O,P)) # matrix de zeros com dimensões (NxM) 
    
    scalar_vec_real = {
        'dumb': scalar_vec_real_dumb,
        'numpy': scalar_vec_real_numpy,
        'numba': scalar_vec_real_numba
    }
    if function not in scalar_vec_real:
        raise ValueError("Function {} not recognized".format(function)) # (x.real[i],y.real[:])
        
    # tem que ter 1 for, deve dar certo:
    for i in range(O):
        Q[i][:] = scalar_vec_real[function](x.real[i],y.real[:])# automatizando com (:) não precisa do for!
            
    return Q


# Pelo método numba:

@njit
def outer_real_numba(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.
    The code uses numba.
    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array'
        assert y.ndim == 1, 'y must be a 1D array'
        
    O = np.size(x) # dimensão do vetor x
    P = np.size(y) # dimensão do vetor y
    Q = np.zeros((O,P)) # matrix de zeros com dimensões (NxM) 
    
    for i in range (O):
        for j in range (P):
            Q[i][j] = x.real[i]*y.real[j]
            
    return Q


# Operação vet x vet = matrix, com vetor complexo:

def outer_complex(x, y, check_input=True,  function='numpy'):
    '''
    Compute the outer product of x and y, where x and y are complex vectors.
    Parameters
    ----------
    x, y : 1D arrays
        Complex vectors.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real outer product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numpy'.
    Returns
    -------
    result : 2D array
        Outer product of x and y.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array'
        assert y.ndim == 1, 'y must be a 1D array'
        
    outer_vec_real = {
        'dumb': outer_real_dumb,
        'numpy': outer_real_numpy,
        'numba': outer_real_numba
    }
    if function not in outer_vec_real:
        raise ValueError("Function {} not recognized".format(function))
        
    result_real = outer_vec_real[function](x.real, y.real, check_input=False)
    result_real -= outer_vec_real[function](x.imag, y.imag, check_input=False)
    result_imag = outer_vec_real[function](x.real, y.imag, check_input=False)
    result_imag += outer_vec_real[function](x.imag, y.real, check_input=False)
    
    result = result_real + 1j*result_imag
    
    return result


###################################################### Cross product ###########################################################

def cross_real_dumb(x, y, check_input=True):
    '''
    Cross product between two vectors
    
    input:
    x,y = vector 1D with 3 elements
    
    output:
    s = cross product (determinante)
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array'
        assert x.size == 3, 'must be a vector with 3 elements' 
        assert x.size == y.size, 'x and y must be same shape'
        
    s = np.zeros_like(x) 
    s = (x.real[1]*y.real[2] - y.real[1]*x.real[2]), -(x.real[0]*y.real[2] - y.real[0]*x.real[2]), (x.real[0]*y[1] - y.real[0]*x[1])
    
    return s


@njit
def cross_real_numba(x, y, check_input=True):
    '''
    Cross product between two vectors
    Obs: using numba!
    
    input:
    x,y = vector 1D with 3 elements
    
    output:
    s = cross product (determinante)
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)
    if check_input is True:
        assert x.ndim == 1, 'x must be a 1D array'
        assert x.size == 3, 'must be a vector with 3 elements' 
        assert x.size == y.size, 'x and y must be same shape'
        
    s = np.zeros_like(x) 
    s = (x.real[1]*y.real[2] - y.real[1]*x.real[2]), -(x.real[0]*y.real[2] - y.real[0]*x.real[2]), (x.real[0]*y[1] - y.real[0]*x[1])
    
    return s


######################################################## SMA ################################################################

def mat_sma(vec, win, check_input = True):
    '''
    This function calculates the SMA given a vector of data, 
    using a given window of size equal to N.
    
    Input:
    vec = 1D array
    win = integer constant
    
    Output:
    vec_out =  calculated vector
    '''
    if check_input is True:
        assert vec.size >= win, 'Vector must have more elements than window.'
        assert win%2 != 0, 'Window must be different.'
        assert win >= 3, 'Window size increment.'

    N = vec.size
    devWin = win//2
    
    matA = np.array(np.hstack(((1./win)*np.ones(win), np.zeros(N-win+1)))) #np.hstack = Empilhe os arrays em sequência horizontalmente (coluna)
    matA = np.resize(matA, (N-2*devWin, N)) #Devolver uma nova matriz com a forma especificada
    matA = np.vstack((np.zeros(N), matA, np.zeros(N))) #np.vstack = Empilhe os arrays em sequência verticalmente (linha)
    
    vec_out = np.zeros([N,1])
    vec_out = matvec_prod1(matA,vec)
    
    return vec_out


#################################################### Matrix-vector product #####################################################

# matrix-vector dumb:

def matvec_real_dumb(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.
    The code uses a simple doubly nested "for" to iterate on the arrays.
    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.
    x : array 1D
        Real vector witn M elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    
    A = np.asarray(A)
    x = np.asarray(x)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert x.ndim == 1, 'x must be a 1D array'
        assert A.shape[1] == x.size, 'column A must be = row x' # x.shape[0],
    
    N, M = np.shape(A) # N e M são as dimensões de lihas e colunas
    y = np.zeros(N) # vetor vazio que serão preenchidos no laço abaixo # A.real[i,j] * x.real[j]
    
        
    for i in range(N):
        for j in range(M):
            y[i] += A.real[i,j] * x.real[j]
            
    return y   


# matrix-vector numba:

@njit
def matvec_real_numba(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.
    The code uses numba jit.
    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.
    x : array 1D
        Real vector witn M elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    
    A = np.asarray(A)
    x = np.asarray(x)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert x.ndim == 1, 'x must be a 1D array'
        assert A.shape[1] == x.size, 'column A must be = row x' # x.shape[0],
    
    N, M = np.shape(A) # N e M são as dimensões de lihas e colunas
    y = np.zeros(N) # vetor vazio que serão preenchidos no laço abaixo # A.real[i,j] * x.real[j]
    
        
    for i in range(N):
        for j in range(M):
            y[i] += A.real[i,j] * x.real[j]
            
    return y


# matrix-vector dot:

def matvec_real_dot(A, x, check_input=True,function='numpy'):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.
    The code replaces a for by a dot product.
    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.
    x : array 1D
        Real vector witn M elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real dot product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numpy'.
    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    
    A = np.asarray(A)
    x = np.asarray(x)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert x.ndim == 1, 'x must be a 1D array'
        assert A.shape[1] == x.size, 'column A must be = row x'
        
    N, M = np.shape(A) # N e M são as dimensões de lihas e colunas
    y = np.zeros(N) # vetor vazio que serão preenchidos no laço abaixo A[i,:], x[:]
    
    dot_real = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'numba':dot_real_numba,
        'complex': dot_complex
    }
    if function not in dot_real:
        raise ValueError("Function {} not recognized".format(function)) #[function](A.real[i,:], x.real[:])
        
    for i in range(N):
        y[i] += dot_real[function](A.real[i,:], x.real[:])
        
    return y  


# matrix-vector columns:

def matvec_real_columns(A, x, check_input=True, function='numpy'):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.
    The code replaces a for by a scalar-vector product.
    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.
    x : array 1D
        Real vector witn M elements.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real dot product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'numpy'.
    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    
    A = np.asarray(A)
    x = np.asarray(x)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert x.ndim == 1, 'x must be a 1D array'
        assert A.shape[1] == x.size, 'column A must be = row x'
    
    scalar_vec_real = {
        'dumb': scalar_vec_real_dumb,
        'numpy': scalar_vec_real_numpy,
        'numba': scalar_vec_real_numba
    }
    if function not in scalar_vec_real:
        raise ValueError("Function {} not recognized".format(function)) # (x.real[j], A.real[:,j])
        
    N, M = np.shape(A) # N e M são as dimensões de lihas e colunas
    y = np.zeros(N) # vetor vazio que serão preenchidos no laço abaixo x[j], A[:,j]  
    
    for j in range(M):
        y += scalar_vec_real[function](x.real[j], A.real[:,j], check_input=False)
                              
    return y


# matrix-vector complex:

def matvec_complex(A, x, check_input=True, function='dumb'):
    '''
    Compute the matrix-vector product of an NxM matrix A and
    a Mx1 vector x.
    Parameters
    ----------
    A : array 2D
        NxM matrix.
    x : array 1D
        Mx1 vector.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real mattrix-vectorvec product.
        The function name must be 'dumb', 'numba', 'dot' or 'columns'.
        Default is 'dumb'.
    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    
    A = np.asarray(A)
    x = np.asarray(x)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert x.ndim == 1, 'x must be a 1D array'
        assert A.shape[1] == x.size, 'column A must be = row x'

    matvec_real = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba
    }
    if function not in matvec_real:
        raise ValueError("Function {} not recognized".format(function))

    y_R = matvec_real[function](A.real, x.real, check_input=False)
    y_R -= matvec_real[function](A.imag, x.imag, check_input=False)
    y_Im = matvec_real[function](A.real, x.imag, check_input=False)
    y_Im += matvec_real[function](A.imag, x.real, check_input=False)

    result = y_R + 1j*y_Im

    return result


##################################################### Matrix-matrix product ####################################################

# Matrix-matrix dumb:

def matmat_real_dumb(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.
    The code uses a simple triply nested "for" to iterate on the arrays.
    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    
    A = np.asarray(A)
    B = np.asarray(B)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D array (matrix)'
        assert B.ndim == 2, 'B must be a 2D array (matrix)'
        assert A.shape[1] == B.shape[0], 'column A must be = row x'
        
    N, M = np.shape(A) # N e M são as dimensões de lihas e colunas
    M, L = np.shape(B) #
    C = np.zeros((N,L)) # vetor vazio que serão preenchidos no laço abaixo C[i,j] += A[i,k] * B[k,j]
    
    for i in range(N):
        for j in range(L):
            for k in range(M):
                C[i,j] += A.real[i,k] * B.real[k,j]
    return C 


# Matrix-matrix numba:
@njit
def matmat_real_numba(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.
    The code uses numba.
    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    
    A = np.asarray(A)
    B = np.asarray(B)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D array (matrix)'
        assert B.ndim == 2, 'B must be a 2D array (matrix)'
        assert A.shape[1] == B.shape[0], 'column A must be = row x'
        
    N, M = np.shape(A) # N e M são as dimensões de lihas e colunas
    M, L = np.shape(B) #
    C = np.zeros((N,L)) # vetor vazio que serão preenchidos no laço abaixo C[i,j] += A[i,k] * B[k,j]
    
    for i in range(N):
        for j in range(L):
            for k in range(M):
                C[i,j] += A.real[i,k] * B.real[k,j]
    return C 


# Matrix-matrix dot:

def matmat_real_dot(A, B, check_input=True, function='numpy'):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.
    The code replaces one "for" by a dot product.
    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real dot product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'dumb'.
    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    
    A = np.asarray(A)
    B = np.asarray(B)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D array (matrix)'
        assert B.ndim == 2, 'B must be a 2D array (matrix)'
        assert A.shape[1] == B.shape[0], 'column A must be = row x'
        
    N, M = np.shape(A) # N e M são as dimensões de lihas e colunas
    M, L = np.shape(B) #
    C = np.zeros((N,L)) # vetor vazio que serão preenchidos no laço abaixo
    
    dot_vec_real = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'numba': dot_real_numba
    }
    if function not in dot_vec_real:
        raise ValueError("Function {} not recognized".format(function)) # dot_vec_real[function](A[i,:], B[:,j])
    
    for i in range(N):
        for j in range(L):
            C[i,j] += dot_vec_real[function](A.real[i,:], B.real[:,j])
            
    return C 


# Matrix-matrix columns:

def matmat_real_columns(A, B, check_input=True, function='dumb'):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.
    The code replaces two "fors" by a matrix-vector product defining
    a column of the resultant matrix.
    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real matrix-vector product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'dumb'.
    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    
    A = np.asarray(A)
    B = np.asarray(B)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D array (matrix)'
        assert B.ndim == 2, 'B must be a 2D array (matrix)'
        assert A.shape[1] == B.shape[0], 'column A must be = row x'
        
    N, M = np.shape(A) # N e M são as dimensões de lihas e colunas
    M, L = np.shape(B) #
    C = np.zeros((N,L)) # vetor vazio que serão preenchidos no laço abaixo
    
    scalar_real = {
        'dumb': scalar_vec_real_dumb,
        'numpy': scalar_vec_real_numpy,
        'numba': scalar_vec_real_numba,
        'complex': scalar_vec_complex
    }
    if function not in scalar_real:
        raise ValueError("Function {} not recognized".format(function)) # (A[:,:], B[:,j])
        
    for j in range(L):
        for k in range(M):
            C[:,j] += scalar_real[function](B.real[k,j], A.real[:,k], check_input=False) # Com escalar
        
    return C 


# Matrix-matrix outer:

def matmat_real_outer(A, B, check_input=True, function='dumb'):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.
    The code replaces two "for" by an outer product.
    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real outer product.
        The function name must be 'dumb', 'numpy' or 'numba'.
        Default is 'dumb'.
    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    
    A = np.asarray(A)
    B = np.asarray(B)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D array (matrix)'
        assert B.ndim == 2, 'B must be a 2D array (matrix)'
        assert A.shape[1] == B.shape[0], 'column A must be = row x'
        
    N, M = np.shape(A) # N e M são as dimensões de lihas e colunas
    M, L = np.shape(B) #
    C = np.zeros((N,L)) # vetor vazio que serão preenchidos no laço abaixo
    
    outer_vec_real = {
        'dumb': outer_real_dumb,
        'numpy': outer_real_numpy,
        'numba': outer_real_numba,
        'complex': outer_complex
    }
        
    if function not in outer_vec_real:
        raise ValueError("Function {} not recognized".format(function)) # (A[:,k], B[k,:])
        
    for k in range(M):
        C[:,:] += outer_vec_real[function](A.real[:,k], B.real[k,:])
        
    return C


# Matrix-matrix matvec:

def matmat_real_matvec(A, B, check_input=True, function='dot'):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.
    The code replaces two "fors" by a matrix-vector product defining
    a row of the resultant matrix.
    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real matrix-vector product.
        The function name must be 'dumb', 'numba', 'dot' and 'columns'.
        Default is 'dumb'.
    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    
    A = np.asarray(A)
    B = np.asarray(B)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D array (matrix)'
        assert B.ndim == 2, 'B must be a 2D array (matrix)'
        assert A.shape[1] == B.shape[0], 'column A must be = row y'
        
    N, M = np.shape(A) # N e M são as dimensões de lihas e colunas
    M, L = np.shape(B) #
    C = np.zeros((N,L)) # vetor vazio que serão preenchidos no laço abaixo
    
    matvec_real = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'complex': matvec_complex
    }
    if function not in matvec_real:
        raise ValueError("Function {} not recognized".format(function))
        
    for j in range(L):
        C[:,j] = matvec_real[function](A.real[:,:], B.real[:,j])
        
    return C


# Matrix-matrix complex:

def matmat_complex(A, B, check_input=True, function='dot'):
    '''
    Compute the matrix-matrix product of A and B, where
    A in C^NxM and B in C^MxP.
    Parameters
    ----------
    A, B : 2D arrays
        Complex matrices.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    function : string
        Function to be used for computing the real and imaginary parts.
        The function name must be 'dumb', 'numba', 'dot', 'columns', 'matvec'
        and 'outer'. Default is 'dumb'.
    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
        
    A = np.asarray(A)
    B = np.asarray(B)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D array (matrix)'
        assert B.ndim == 2, 'B must be a 2D array (matrix)'
        assert A.shape[1] == B.shape[0], 'column A must be = row x'

    matmat_real = {
        'dumb': matmat_real_dumb,
        'dot': matmat_real_dot,
        'columns': matmat_real_columns,
        'outer': matmat_real_outer,
        'matvec': matmat_real_matvec,
        'numba': matmat_real_numba
        
    }
    if function not in matmat_real:
        raise ValueError("Function {} not recognized".format(function))

    C_R = matmat_real[function](A.real, B.real, check_input=False)
    C_R -= matmat_real[function](A.imag, B.imag, check_input=False)
    C_Im = matmat_real[function](A.real, B.imag, check_input=False)
    C_Im += matmat_real[function](A.imag, B.real, check_input=False)

    result = C_R + 1j*C_Im

    return result


###################################################### Derivada 1D #######################################################

def deriv1d(vec, h, check_input=True, function='dumb'):
    '''
    This function calculates the first derivative of a vector.

    Input:
    vec = 1D array
    h = constant
    
    Output:
    return dvec (first derivative)
    '''
    vec = np.asarray(vec)
    h = np.asarray(h)
    if check_input is True:
        assert vec.ndim == 1, 'vec must be a 1D array (vector)'
        assert h.ndim == 0, 'h must be a constant'
    
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    if function not in matvec:
        raise ValueError("Function {} not recognized".format(function))
        
    n = vec.size
    lost = 3//2
    
    matD = np.array(np.hstack(((np.array([-1., 0., 1.]), np.zeros(n-3+1)))))
    matD = np.resize(matD, (n-2*lost,n))
    matD = np.vstack((np.zeros(n), matD, np.zeros(n)))
    matD = (1./(2*h))*matD
    
    dvec = matvec[function](matD,vec)
    
    return dvec


###################################################### Triangular matrices ####################################################

# algoritmo 3 triangular superior:

def matvec_triu_prod3(A, x, check_input=True, function='dumb'):
    '''
    input:
    A = 2D array (matrix)
    x = 1D array (vector)
    
    Operation:
    U = uper triangle matrix from A
    -> using dot from function dot_real_dumb
    
    output:
    y = 1D array (vector) -> (A.x)
    '''
    
    A = np.asarray(A)
    x = np.asarray(x)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert x.ndim == 1, 'x must be a 1D array'
        assert A.shape[1] == x.shape[0], 'column A must be = row x' # A.shape[0] n de linhas [1] n de colunas
        assert A.shape[1] == A.shape[0], 'Must be square matrix'
    
    N, M = np.shape(A)
    L = np.size(x)
    y = np.zeros(N)
    
    dot_real = {
        'dumb': dot_real_dumb,
        'numba': dot_real_numba,
        'numpy': dot_real_numpy,
        'complex': dot_complex,
    }
    if function not in dot_real:
        raise ValueError("Function {} not recognized".format(function)) 
        
    for i in range(N):
        y[i] = dot_real[function](A.real[i,i:], x.real[i:])
        
    return y


# algoritmo 5 triangular superior:

def matvec_triu_prod5(A, x, check_input=True):
    '''
    input:
    A = 2D array (matrix)
    x = 1D array (vector)
    
    Operation:
    U = uper triangle matrix from A
    -> using dot from function dot_real_dumb
    
    output:
    y = 1D array (vector) -> (A.x)
    
    '''
    A = np.asarray(A)
    x = np.asarray(x)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert x.ndim == 1, 'x must be a 1D array'
        assert A.shape[1] == x.shape[0], 'column A must be = row x' # A.shape[0] n de linhas [1] n de colunas
        assert A.shape[1] == A.shape[0], 'Must be square matrix'
    
    N, M = np.shape(A)
    L = np.size(x)
    y = np.zeros(N)
    
    for j in range(N):          #  A[:j+1,j] * x[j]
        y[:j+1] += A.real[:j+1,j] * x.real[j]
        
    return y


# algoritmo 8 triangular inferior:

def matvec_tril_prod8(A, x, check_input=True, function='dumb'):
    '''
    input:
    A = 2D array (matrix)
    x = 1D array (vector)
    
    Operation:
    L = lower triangle matrix from A
    -> using dot from function dot_real_dumb
    
    output:
    y = 1D array (vector) -> (A.x)
    '''
    N, M = np.shape(A)
    y = np.zeros(N)
    
    A = np.asarray(A)
    x = np.asarray(x)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert x.ndim == 1, 'x must be a 1D array'
        assert A.shape[1] == x.shape[0], 'column A must be = row x'
        assert A.shape[1] == A.shape[0], 'Must be square matrix'
        
    dot_real = {
        'dumb': dot_real_dumb,
        'numba': dot_real_numba,
        'numpy': dot_real_numpy,
        'complex': dot_complex,
    }
    
    if function not in dot_real:
        raise ValueError("Function {} not recognized".format(function)) # (A[i,:i+1], x[:i+1])
        
    for i in range(N):
        y[i] = dot_real[function](A.real[i,:i+1], x.real[:i+1])
        
    return y


# algoritmo 10 triangular inferior:

def matvec_tril_prod10(A, x, check_input=True):
    '''
    input:
    A = 2D array (matrix)
    x = 1D array (vector)
    
    Operation:
    L = lower triangle matrix from A
    -> using dot from function dot_real_dumb
    
    output:
    y = 1D array (vector) -> (A.x)
    
    '''
    N, M = np.shape(A)
    y = np.zeros(N)
    
    A = np.asarray(A)
    x = np.asarray(x)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert x.ndim == 1, 'x must be a 1D array'
        assert A.shape[1] == x.shape[0], 'column A must be = row x'
        assert A.shape[1] == A.shape[0], 'Must be square matrix'
    
    for j in range(N):
        y[j:] += A.real[j:,j] * x.real[j]
        
    return y


##################################################### Triangular systems ######################################################

# tringular superior:

def triu_system(A, y, check_input=True, function='dumb'):
    '''
    Function to solve a upper triangular system
    
    input:
    A = 2D array (matrix)
    y = 1D array (vector)
    
    Operation:
    U = upper triangle matrix from A
    -> using dot from function "dot_real"
    
    output:
    return
    x = 1D array (vector) for A.x = y solution 
    '''
    
    N, M = np.shape(A)
    x = np.zeros(N)
    
    A = np.asarray(A)
    y = np.asarray(y)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert y.ndim == 1, 'y must be a 1D array'
        assert A.shape[1] == y.shape[0], 'column A must be = rows y'
        assert A.shape[1] == A.shape[0], 'Must be square matrix'
    
    dot_real = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'complex': dot_complex
    }
    if function not in dot_real:
        raise ValueError("Function {} not recognized".format(function)) # (A[i,i+1:],x[i+1:])
    
    for i in range(N-1,0-1,-1):
        x[i] = y.real[i] - dot_real[function](A.real[i,i+1:],x.real[i+1:])
        x[i] = x.real[i]/A.real[i,i]
        
    return x


# triangulo inferior:

def tril_system(A, y, check_input=True, function='dumb'):
    '''
    Function to solve a lower triangular system
    
    input:
    A = 2D array (matrix)
    y = 1D array (vector)
    
    Operation:
    L = lower triangle matrix from A
    -> using dot from function "dot_real"
    
    output:
    return
    x = 1D array (vector) for A.x = y solution 
    '''
    
    N, M = np.shape(A)
    x = np.zeros(N)
    
    A = np.asarray(A)
    y = np.asarray(y)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert y.ndim == 1, 'y must be a 1D array'
        assert A.shape[1] == y.shape[0], 'column A must be = rows y'
        assert A.shape[1] == A.shape[0], 'Must be square matrix'
    
    dot_real = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'complex': dot_complex,
    }
    if function not in dot_real:
        raise ValueError("Function {} not recognized".format(function)) # (A[i,:i],x[:i])
        
    for i in range(N):
        x[i] = y.real[i] - dot_real[function](A.real[i,:i],x.real[:i])
        x[i] = x.real[i]/A.real[i,i]
        
    return x


##################################################### Gauss elimination ######################################################

# Solução de sistema pelo método Gaussiano (EXTRA):

def Gauss_elim_dumb(A, y, check_input=True, function='dumb'):
    '''
    Gaussian elimination for square matrix
    Obs: extra function where the output returns 
    an upper triangular matrix (equivalent triangle system)
    
    Input:
    A = 2D array (matrix)
    y = 1D array (vector)
    
    Operation:
    y = Ax 
    Usin "outer product"
    
    Output:
    return A and y altered
    '''
    
    N, M = np.shape(A) # armazenar o número de linhas e colunas
    x = np.zeros(N) # vetor que receberá os resultados da solução Gaussiana
    I =  np.identity(N) # matriz identidade com tamanho N
    
    A = np.asarray(A)
    y = np.asarray(y)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert y.ndim == 1, 'y must be a 1D array'
        assert A.shape[1] == y.shape[0], 'column A must be = row y'
        assert A.shape[1] == A.shape[0], 'Must be square matrix'
    
    outer_vec = {
        'dumb': outer_real_dumb,
        'dot': outer_real_numpy,
        'numba': outer_real_numba,
        'complex': outer_complex
    }
    
    if function not in outer_vec:
        raise ValueError("Function {} not recognized".format(function)) 
    
    for i in range(N-1):
        # u
        u0 = np.zeros(N) # criação do vetor u0
        u0[i] = 1
        #print('u0=', u0)
        # t
        t = np.zeros(N) # criação do vetor t
        t[i+1:] = A[i+1:,i]/A[i,i] # Vetor de Gauss
        #print('t=', t)
        # A e y usando as funções "outer" neste template:
        A = (I - outer_vec[function](t, u0))@A # Transformação de Gauss para A
        #print('A=', A)
        y = (I - outer_vec[function](t, u0))@y # Transformação de Gauss para y
        
    # solução x usando a função "triu_system" neste template:  
    #x = triu_system(A, y)
    
    return A, y


# Função para Permutação:

def permut (C, i):
    p = [j for j in range(C.shape[0])]
    imax = i + np.argmax(np.abs(C[i:,i]))
    if imax != i:
        p[i], p[imax] = p[imax], p[i]
    return p, C[p,:]


# Função de eliminação Gaussiana com matrix C (matrix expandida):

def Gauss_elim(A, y, check_input=True, function='dumb'):
    '''
    Gaussian elimination by the "outer product" method
    Obs: function where the output returns an upper 
    triangular matrix with the Gaussian multipliers below the main diagonal
    
    Input:
    A = 2D array (matrix)
    y = 1D array (vector)
    
    Operation:
    -> C = [A|y] stacking A and y 
    -> C Permutation
    -> Gauss multipliers 
    -> Outer products for C
    
    Output:
    return C = [A|y] updated
    '''
    
    N = A.shape[0] # armazenar o número de linhas
    
    A = np.asarray(A)
    y = np.asarray(y)
    if check_input is True:
        assert A.ndim == 2, 'A must be a 2D matrix'
        assert y.ndim == 1, 'y must be a 1D array'
        assert A.shape[1] == N, 'A must be square'
        assert y.size == N, 'A columns must be equal to y size'
            
    outer_vec = {
        'dumb': outer_real_dumb,
        'dot': outer_real_numpy,
        'numba': outer_real_numba,
        'complex': outer_complex
    }
    if function not in outer_vec:
        raise ValueError("Function {} not recognized".format(function)) 
    
    # create matrix C by stacking A and y
    C = np.vstack([A.T, y]).T
    
    for i in range(N-1):
        # permutation step (computation of C tilde - eq. 3)
        p, C = permut(C, i)

        # assert the pivot is nonzero
        assert C[i,i] != 0., 'null pivot!'

        # calculate the Gauss multipliers and store them 
        # in the lower part of C (equations 5 and 7)
        C[i+1:,i] = C[i+1:,i]/C[i,i]

        # zeroing of the elements in the (k-1)th column (equation 8)
        C[i+1:,i+1:] -= outer_vec[function](C[i+1:,i],C[i,i+1:])

    # return the equivalent triangular system and Gauss multipliers
    return C[:,:N], C[:,N]


# Função de eliminação Gaussiana com matrix C (matrix expandida) com uma matrix Y de entrada:

def Gauss_elim_expanded(A, Y, check_input=True, function='dumb'):
    '''
    Gaussian elimination by the "outer product" method
    Obs: If matrix Y is the identity, the process returns the inverse matrix A
    
    Input:
    A = 2D array (matrix)
    Y = 2D array (matrix), with the vectors forming the columns of Y.
    
    Operation:
    -> C = [A|Y] stacking A and Y 
    -> C Permutation
    -> Gauss multipliers 
    -> Outer products for C
    
    Output:
    return C = [A|Y] updated
    '''
    
    N = A.shape[0] # armazenar o número de linhas
    #I = np.identity(N) # criar a matrix identidade com tamanho N
    
    A = np.asarray(A)
    Y = np.asarray(Y)
    if check_input is True:
        assert A.ndim == Y.ndim == 2, 'A and Y must be matrices'
        assert A.shape[1] == N, 'A must be square'
        assert Y.shape[0] == N, 'A columns must have the same size as Y rows'
            
    outer_vec = {
        'dumb': outer_real_dumb,
        'dot': outer_real_numpy,
        'numba': outer_real_numba,
        'complex': outer_complex
    }
    if function not in outer_vec:
        raise ValueError("Function {} not recognized".format(function)) 
        
    C = np.vstack([A.T, Y]).T
    
    for i in range(N-1):
        # permutation step (computation of C tilde - eq. 3)
        p, C = permut(C, i)

        # assert the pivot is nonzero
        assert C[i,i] != 0., 'null pivot!'

        # calculate the Gauss multipliers and store them 
        # in the lower part of C (equations 5 and 7)
        C[i+1:,i] = C[i+1:,i]/C[i,i]

        # zeroing of the elements in the (k-1)th column (equation 8)
        C[i+1:,i+1:] -= outer_vec[function](C[i+1:,i],C[i,i+1:])

    # return the equivalent triangular system and Gauss multipliers
    return C[:,:N], C[:,N:]


# Função para solução do Gauss_elim_expanded:

def Gauss_solution_inv(B, Z, check_input=True):
    '''
    Function for stacking Matrix B with vectors Z[i],
    and calculate the inverse matrix.
    
    Obs: for the function Gauss_elim_expanded
    
    input:
    B = 2D array (matrix)
    Z = 2D array (matrix)
    
    Operation:
    triu_system function solving by column and staking in A_inv matrix
    
    Output:
    Solution for inverse matrix (A_inv)
    '''
    
    M = B.shape[0]
    
    B = np.asarray(B)
    Z = np.asarray(Z)
    if check_input is True:
        assert B.ndim == Z.ndim == 2, 'B and Z must be matrices'
        assert B.shape[1] == M, 'B must be square'
        assert Z.shape[0] == M, 'B columns must have the same size as Z rows'
        
    A_inv = np.zeros((M,M))
    W = np.zeros((M,M))

    for i in range(M):
        W[i] = triu_system(B, Z[:,i])
        A_inv[:,i] = np.vstack([W[i]])
    
    return A_inv


##################################################### LU decomposition ######################################################

# Decomposição LU:

def lu_decomp(A, check_input=True, function='dumb'):
    '''
    LU decomposition from input "A"
    Obs: extra function where the output returns an upper 
    triangular matrix with the Gaussian multipliers below the main diagonal
    
    Input:
    A = 2D array (matrix)
    
    Operation:
    -> C = [A] stacking C = copy (A) 
    -> Gauss multipliers 
    -> Outer products for C
    
    Output:
    return C = "L+U" updated
    '''
    
    N = A.shape[0] # armazenar o número de linhas
    
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
            
    outer_vec = {
        'dumb': outer_real_dumb,
        'dot': outer_real_numpy,
        'numba': outer_real_numba,
        'complex': outer_complex
    }
    if function not in outer_vec:
        raise ValueError("Function {} not recognized".format(function)) 
    
    # create matrix C copying A
    C = np.copy((A))
    
    for i in range(N-1):

        # assert the pivot is nonzero
        assert C[i,i] != 0., 'null pivot!'

        # calculate the Gauss multipliers and store them 
        # in the lower part of C (equations 5 and 7)
        C[i+1:,i] = C[i+1:,i]/C[i,i]

        # zeroing of the elements in the (k-1)th column (equation 8)
        C[i+1:,i+1:] -= outer_vec[function](C[i+1:,i],C[i,i+1:])
        #C[i+1:,i+1:] = C[i+1:,i+1:] - outer_vec[function](C[i+1:,i],C[i,i+1:])
    
    # return the equivalent triangular system and Gauss multipliers
    return C


# Decomposição LU com saida com L e U separados (EXTRA):

def lu_decomp_alt(A, check_input=True, function='dumb'):
    '''
    LU decomposition from input "A"
    Obs: this extra function returns L and U separately (altered - alt)
    
    Input:
    A = 2D array (matrix)
    
    Operation:
    -> C = [A] stacking C = copy (A) 
    -> C Permutation
    -> Gauss multipliers 
    -> Outer products for C
    
    Output:
    return L and U updated separated
    '''
    
    N = A.shape[0] # armazenar o número de linhas
    
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
            
    outer_vec = {
        'dumb': outer_real_dumb,
        'dot': outer_real_numpy,
        'numba': outer_real_numba,
        'complex': outer_complex
    }
    if function not in outer_vec:
        raise ValueError("Function {} not recognized".format(function)) 
    
    # create matrix C copying A
    C = np.copy((A))
    L = np.zeros((N,N)) # matrix L
    U = np.zeros((N,N)) # matrix U
    
    for i in range(N-1):

        # assert the pivot is nonzero 
        assert C[i,i] != 0., 'null pivot!'

        # calculate the Gauss multipliers and store them 
        # in the lower part of C (equations 5 and 7)
        C[i+1:,i] = C[i+1:,i]/C[i,i]

        # zeroing of the elements in the (k-1)th column (equation 8)
        C[i+1:,i+1:] -= outer_vec[function](C[i+1:,i],C[i,i+1:])
    
    # criando a matrix L com os elementos da diagonal principal igual a 0 ou 1:
    for j in range(N):
        for k in range(N):
            if j==k:
                L[j,k] = 1
            if j!=k:
                L[j,k] = C[j,k]
    
    L = np.tril(L) # L matrix com os elementos da matrix inferior de C.  
    U = np.triu(C) # U matrix com os elementos da matrix superior de C.
    
    # return the equivalent triangular system and Gauss multipliers
    return L, U


# Solução LU:

def lu_solve(C, y, check_input=True):
    '''
    Function to calculate the solution to 
    "LU" using the "triangular systems" functions
    
    input:
    C = array 2D (matrix) before lu_decomp function
    y = array 1D (vector) for system Ax = y
    
    operation:
    Lw = y solution for w; 
    Ux = w solution for x
    
    output:
    return solution x
    '''
    
    N = C.shape[0] # armazenar o número de linhas
    
    C = np.asarray(C)
    y = np.asarray(y)
    if check_input is True:
        assert C.ndim == 2, 'C must be a matrix'
        assert y.ndim == 1, 'y must be a vector'
        assert C.shape[1] == N, 'C must be square'

    w = np.zeros(N)
    x = np.zeros(N)
    L = np.zeros((N,N)) # matrix L
    U = np.zeros((N,N)) # matrix U
    
    for j in range(N):
        for k in range(N):
            if j==k:
                L[j,k] = 1
            if j!=k:
                L[j,k] = C[j,k]
             
    L = np.tril(L)
    U = np.triu(C)
    
    w = tril_system(L, y)
    x = triu_system(U, w)
    
    #return x # Retorna apenas a solução x
    return L, U, x # se quiser o retorno de L, U e x ao mesmo tempo


# LU com pivoteamento:

def lu_decomp_pivoting(A, check_input=True, function='dumb'):
    '''
    LU decomposition from input "A" with pivoting
    
    Input:
    A = 2D array (matrix)
    
    Operation:
    -> C = [A] stacking C = copy (A) 
    -> C Permutation
    -> Gauss multipliers 
    -> Outer products for C
    
    Output:
    return P and C
    where P is a record of permutations and C altered
    '''
   
    N = A.shape[0]
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'

    # create matrix C as a copy of A
    C = np.copy(A)
    P = np.identity((N)) #list containing elements varying from 0 to N-1

    outer_vec = {
        'dumb': outer_real_dumb,
        'dot': outer_real_numpy,
        'numba': outer_real_numba,
        'complex': outer_complex
    }
    if function not in outer_vec:
        raise ValueError("Function {} not recognized".format(function)) 
        
    for k in range(N-1):

        # permutation step # p, C = permut(C, i)
        p, C = permut(C, k)
        P = P[p]

        # assert the pivot is nonzero
        assert C[k,k] != 0., 'null pivot!'

        # calculate the Gauss multipliers and store them 
        # in the lower part of C
        C[k+1:,k] = C[k+1:,k]/C[k,k]

        # zeroing of the elements in the (k-1)th column
        C[k+1:,k+1:] -= outer_vec[function](C[k+1:,k],C[k,k+1:])

    # return matrix P and C
    return P, C


# Solução LU com pivotamento:

def lu_solve_pivoting(P, C, y, check_input=True, function='dumb'):
    '''
    Function to calculate the solution to 
    "LU" using the "triangular systems" functions
    
    input:
    P = array 2D (matrix) before lu_decomp function (record of permutations)
    C = array 2D (matrix) before lu_decomp function
    y = array 1D (vector) for system Ax = y
    
    operation:
    dot(P, y) for permutation of vector y
    Lw = y solution for w; 
    Ux = w solution for x
    
    output:
    return solution x
    '''
    
    N = C.shape[0] # armazenar o número de linhas
    
    C = np.asarray(C)
    P = np.asarray(P)
    y = np.asarray(y)
    if check_input is True:
        assert C.ndim == 2, 'C must be a matrix'
        assert P.ndim == 2, 'P must be a matrix'
        assert y.ndim == 1, 'y must be a matrix'
        assert y.shape[0] == N, 'y must be a same size C rows'
        assert C.shape[1] == N, 'C must be square'
        assert P.shape[0] ==  P.shape[1] == N, 'P must be square'

    w = np.zeros(N)
    x = np.zeros(N)
    L = np.zeros((N,N)) # matrix L
    U = np.zeros((N,N)) # matrix U
    
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function not in matvec:
        raise ValueError("Function {} not recognized".format(function))
    
    yy = np.zeros(N)
    yy = matvec[function](P, y)
    
    # Laço para gerar a matrix L com a diagonal principal com o valor 1:
    for j in range(N):
        for k in range(N):
            if j==k:
                L[j,k] = 1
            if j!=k:
                L[j,k] = C[j,k]
             
    L = np.tril(L)
    U = np.triu(C)
    
    w = tril_system(L, yy)
    x = triu_system(U, w)
    
    #return x # Retorna apenas a solução x
    return L, U, x # se quiser o retorno de L, U e x ao mesmo tempo


##################################################### LDLt decomposition ######################################################

# Decomposição LUDt:

def ldlt_decomp(A, check_input=True, function1='dumb', function2='dumb'):
    '''
    Function to calculate LDLt decomposition
    
    input:
    A = array 2D (matrix)
    
    operation:
    create vector d which contains the values of the diagonal matrix D
    create matrix L (lower triangular)
    
    output:
    return L and  d
    '''
    
    N = A.shape[0]
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
        assert np.all(A.T == A), 'A must be symmetric'

    L = np.identity((N)) #identity matrix of order N
    d = np.zeros(N) #vector of zeros with N elements
    
    dot_vec = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'numba': dot_real_numba,
        'complex': dot_complex
    }
    
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function1 not in dot_vec:
        raise ValueError("Function {} not recognized".format(function1))
        
    if function2 not in matvec:
        raise ValueError("Function {} not recognized".format(function2))
        
    for j in range(N):

        v = L[j,:j]*d[:j]
        
        d[j] = A[j,j] - dot_vec[function1](L[j,:j], v)

        L[j+1:,j] = (A[j+1:,j] - matvec[function2](L[j+1:,:j],v))/d[j]

    return L, d


# Decomposição LDLt ovwewrite:

def ldlt_decomp_overwrite(A, check_input=True, function1='dumb', function2='dumb'):
    '''
    Function to calculate LDLt decomposition with overwrite
    
    input:
    A = array 2D (matrix)
    
    operation:
    create vector d which contains the values of the diagonal matrix D
    create matrix L (lower triangular)
    
    output:
    return d
    '''
    
    N = A.shape[0]
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
        assert np.all(A.T == A), 'A must be symmetric'
    
    d = np.zeros(N) #vector of zeros with N elements
    
    dot_vec = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'numba': dot_real_numba,
        'complex': dot_complex
    }
    
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function1 not in dot_vec:
        raise ValueError("Function {} not recognized".format(function1))
        
    if function2 not in matvec:
        raise ValueError("Function {} not recognized".format(function2))
        
    for j in range(N):
        
        v = A[j,:j]*d[:j]
        
        d[j] = A[j,j] - dot_vec[function1](A[j,:j], v)

        A[j+1:,j] = (A[j+1:,j] - matvec[function2](A[j+1:,:j],v))/d[j]

    return d


# Solução LDLt:

def ldlt_solve(L, d, y, check_input=True):
    '''
    Solution of LDLt before LDLt decomposition
    Obs: The matrix L and vector d should be the output of the ldlt_decomp function
    
    input:
    L = array 2D 
    d = array 1D
    y = array 1D (for Ax = y system)
    
    operation:
    operation Lw = y, Dw = z and L.Tx = w using triu and tril system functions
    where A = LDL.T
    
    output:
    return x (solution)
    '''
    
    N = L.shape[0]
    L = np.asarray(L)
    d = np.asarray(d)
    y = np.asarray(y)
    if check_input is True:
        assert L.ndim == 2, 'L must be a matrix'
        assert d.ndim == 1, 'd must be a vector'
        assert y.ndim == 1, 'y must be a vector'
        assert y.shape[0] == N, 'y must be same size L rows'
        assert d.shape[0] == N, 'd must be same size L rows'
        assert L.shape[1] == N, 'L must be square'

    D = np.diag(d)
    
    w = tril_system(L, y) # L*w = y
    z = triu_system(D, w) #D*z = w
    x = triu_system(L.T, z) # L.T*x = z

    return x


# Inversa de A com LDLt:

def ldlt_inverse(A, check_input=True):
    '''
    Function for stacking Matrix B with vectors Z[i],
    and calculate the inverse matrix.
    
    Obs: for output of function Gauss_elim_expanded
    
    input:
    A = 2D array (matrix)
    
    Operation:
    L, d = ldlt decomposition for A
    W = ldlt solve for L, d and column of identity I
    A inverse  = vstack with columns of W
    
    Output:
    Solution for inverse matrix (A_inv)
    '''
    
    N = A.shape[0]
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
        assert np.all(A.T == A), 'A must be symmetric'
        
    I = np.identity((N))
    A_inv = np.zeros((N,N))
    W = np.zeros((N,N))

    for i in range(N):
        L, d = ldlt_decomp(A)
        W[i] = ldlt_solve(L, d, I[:,i])
        A_inv[:,i] = np.vstack([W[i]])

    return A_inv


##################################################### Cholesky decomposition ######################################################

# Decomposição Cholesky:

def cho_decomp(A, check_input=True, function1='dumb', function2='dumb'):
    '''
    Function to calculate Cholesky decomposition
    Obs:The input matrix A must be symmetric
    
    input:
    A = array 2D (matrix)
    
    operation:
    creation of the matrix G using matrix A
    
    output:
    return G
    '''
    
    N = A.shape[0]
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
        assert np.all(A.T == A), 'A must be symmetric'

    G = np.zeros((N,N)) 
    
    dot_vec = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'numba': dot_real_numba,
        'complex': dot_complex
    }
    
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function1 not in dot_vec:
        raise ValueError("Function {} not recognized".format(function1))
        
    if function2 not in matvec:
        raise ValueError("Function {} not recognized".format(function2))

    for j in range(N):

        G[j,j] = A[j,j] - dot_vec[function1](G[j,:j],G[j,:j])

        G[j,j] = np.sqrt(G[j,j])

        G[j+1:,j] = (A[j+1:,j] - matvec[function2](G[j+1:,:j], G[j,:j]))/G[j,j]

    return G


# Decomposição Cholesky overwrite:

def cho_decomp_overwrite(A, check_input=True, function='dumb'):
    '''
    Function to calculate Cholesky decomposition (overwrite)
    Obs:The input matrix A must be symmetric
    
    input:
    A = array 2D (matrix)
    
    operation:
    Cholesky operation using matrix A (input)
    
    output:
    return A altered
    '''
    
    N = A.shape[0]
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
        assert np.all(A.T == A), 'A must be symmetric'
    
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function not in matvec:
        raise ValueError("Function {} not recognized".format(function))
        
    for j in range(N):
        if j > 0:
            A[j:,j] = A[j:,j] - matvec[function](A[j:,:j], A[j,:j])
        A[:,j] = A[:,j]/np.sqrt(A[j,j])
        
    return A


# Solução para Cholesky: cho_solve

def cho_solve(G, y, check_input=True):
    '''
    Solution for Cholesky decomposition
    Obs: The matrix G should be the output of the cho_decomp function
    
    input:
    G = array 2D (matrix)
    y = array 1D (vector)
    
    operation:
    G = lower triangular matrix -> Gw = y
    G.T = upper triangular matrix -> G.Tx = w
    where A  = GG.T
    
    output:
    return x solution
    '''
    
    N = G.shape[0]
    G = np.asarray(G)
    y = np.asarray(y)
    if check_input is True:
        assert G.ndim == 2, 'G must be a matrix'
        assert G.shape[1] == N, 'G must be a square'
        assert y.ndim == 1, 'y must be a vector'
        assert y.shape[0] == N, 'y must be same size G rows'

    w = tril_system(G, y)
    x = triu_system(G.T, w)

    return x


# Inversa de A com Cholesky:

def cho_inverse(A, check_input=True):
    '''
    Function for stacking Matrix B with vectors Z[i],
    and calculate the inverse matrix.
    
    Obs: for the function Gauss_elim_expanded
    
    input:
    A = 2D array (matrix)
    
    Operation:
    G = Cholesky decomposition for A
    W = Cholesky solve for G and column of identity I
    A inverse  = vstack with columns of W
    
    Output:
    Solution for inverse matrix (A_inv)
    '''
    
    N = A.shape[0]
    A = np.asarray(A)
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
        assert np.all(A.T == A), 'A must be symmetric'
        
    I = np.identity((N))
    A_inv = np.zeros((N,N))
    W = np.zeros((N,N))

    for i in range(N):
        G = cho_decomp(A)
        W[i] = cho_solve(G, I[:,i])
        A_inv[:,i] = np.vstack([W[i]])

    return A_inv


######################################################## Least Squares #########################################################

# Mínimos quadrados para y = ax + b:

def straight_line_matrix(x, check_input=True):
    '''
    Assemble the least squares matrix A,
    where the columns are formed by
    the elements that accompany the parameters
    
    input:
    x = 1D array (vector)
    
    operation:
    column[0] = 1
    column[1] = x[i]
    
    output:
    return matrix A (2D array)
    '''
    
    M = np.size(x) # armazena o números de elementos de x
    N = 2 # n° de colunas da matrix A
    A = np.zeros((M,N)) # mAtrix A de zeros M,N
    
    x = np.asarray(x)
    if check_input is True:
        assert x.ndim == 1, 'x must be a vector'
        
    for j in range(N):
        for i in range(M):
            if j == 0: 
                A[i,j] = 1 # atribui o valor 1 para a primeira coluna 
            if j == 1:
                A[i,j] = x[i] # atribui o vetor x na segunda coluna
    
    return A


# solução para o parâmetro p (Ax = y -> x = p):

def straight_line(x, d, check_input=True, function1='dumb', function2='dot'):
    '''
    Function to calculate parameters "a" and "b" using least squares
    obs1: it has to be the same as the inputs "a" and "b"
    obs2: just for equations y = ax + b.
    
    input:
    x = 1D array (vector)
    d = 1D array (vector) data using relation y~d = ax + b
    
    operation:
    At*A*p = At*d where:
    
    At*A = A for relation: Ax = y;
    p = x for relation: Ax = y;
    At*d = y for relation: Ax = y
    
    output:
    return p = a and b (estimated parameters)
    '''
    
    x = np.asarray(x)
    d = np.asarray(d)
    if check_input is True:
        assert x.ndim == 1, 'x must be a vector'
        assert d.ndim == 1, 'x must be a vector'
    
    matmat = {
        'dumb': matmat_real_dumb,
        'dot': matmat_real_dot,
        'columns': matmat_real_columns,
        'outer': matmat_real_outer,
        'matvec': matmat_real_matvec,
        'numba': matmat_real_numba,
        'complex': matmat_complex
        
    }
    if function1 not in matmat:
        raise ValueError("Function {} not recognized".format(function1))
        
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function2 not in matvec:
        raise ValueError("Function {} not recognized".format(function2))
        
    
    A = straight_line_matrix(x)
    
    AtA = matmat[function1](A.T, A)
    
    Atd = matvec[function2](A.T, d)
    
    G,d = Gauss_elim(AtA, Atd)
    
    p = triu_system(G, d)
    
    return p
    
    
# Calculo da covariância:

def parameter_covariance(sig, A, check_input=True, function1='dot', function2='ldlt inv'):
    '''
    Calculate the covariance using my previously implemented functions
    
    input:
    sig = constant
    A = 2D array (matrix)
    
    operation:
    matmat product to calculate AtA
    calculate inverse for AtA
    calculate covariance sig^2 * A inverse
    
    output:
    return W (covariance)
    '''
    
    sig = np.asarray(sig)
    A = np.asarray(A)
    if check_input is True:
        assert sig.ndim == 0, 'sig must be a constant'
        assert A.ndim == 2, 'A must be a matrix'
    
    matmat = {
        'dumb': matmat_real_dumb,
        'dot': matmat_real_dot,
        'columns': matmat_real_columns,
        'outer': matmat_real_outer,
        'matvec': matmat_real_matvec,
        'numba': matmat_real_numba,
        'complex': matmat_complex
        
    }
    
    inverse = { 
        'ldlt inv': ldlt_inverse,
        'Cholesky inv': cho_inverse
    }
    
    if function1 not in matmat:
        raise ValueError("Function {} not recognized".format(function1))
    
    if function2 not in inverse:
        raise ValueError("Function {} not recognized".format(function2))
        
    AtA = matmat[function1](A.T, A)
    
    A_inv = inverse[function2](AtA)
    
    W = (sig**2)*A_inv
    
    return W


############################################# L1 - Fitting a parabola with outliers ###########################################

##### Obs: adaptado com as funções já implementadas (dot, matvec, matmat)!

# Mínimos quadrados comum:

def Ordinary_least_squares_solution(x, data, check_input=True,
                                    function1='dumb', function2='dot'):
    '''
    The least squares will estimate the road parameters
    
    input:
    -> x = 1D array
    -> data = 1D array (data before relation data = ax^2 + bx + c)
    
    operations:
    ->create outliers
    ->create poli-degree 2
    ->create system (A⊤A)p=A⊤d
    ->create y estimated
    
    output:
    return parameters and y estimated
    '''

    xmax = np.max(x)
    xmin = np.min(x)
    Dx = xmax - xmin

    dmin = np.min(data)
    dmax = np.max(data)
    Dd = dmax - dmin
    
    # outlaiers:
    outliers_mask = (x >= 25) & (x <= 28)
    x[outliers_mask]
    data[outliers_mask] -= 8
    
    # polynomial degree = 2 (numpy fuction)
    A = np.polynomial.polynomial.polyvander(x, deg=2)
    
    matmat = {
        'dumb': matmat_real_dumb,
        'dot': matmat_real_dot,
        'columns': matmat_real_columns,
        'outer': matmat_real_outer,
        'matvec': matmat_real_matvec,
        'numba': matmat_real_numba,
        'complex': matmat_complex
        
    }
    
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function1 not in matmat:
        raise ValueError("Function {} not recognized".format(function1))
        
    if function2 not in matvec:
        raise ValueError("Function {} not recognized".format(function2))
    
    # Ordinary least-squares solution
    ATA = matmat[function1](A.T, A)
    ATd = matvec[function2](A.T, data)
    
    G, y = Gauss_elim(ATA, ATd)
    p_LS = triu_system(G, y)
    
    # Ordinary least-squares solution (final equation)
    y_LS = p_LS[0] + p_LS[1]*x + p_LS[2]*x*x
    
    return p_LS, y_LS


# Mínimos quadrados com peso:

def Weighted_least_squares_solution(x, data, nW, check_input=True,
                                    function1='dumb', function2='dot'):
    '''
    The least squares will estimate the road parameters
    obs: with weighted
    
    input:
    -> x = 1D array
    -> data = 1D array (data before relation data = ax^2 + bx + c)
    -> nW = constant (Weighted)
    
    operations:
    ->create outliers
    ->create poli-degree 2
    ->create weighted matrix (W)
    ->create system (A⊤WA)p=A⊤Wd
    ->create y estimated
    
    output:
    return parameters and y estimated
    '''

    xmax = np.max(x)
    xmin = np.min(x)
    Dx = xmax - xmin

    dmin = np.min(data)
    dmax = np.max(data)
    Dd = dmax - dmin
    
    # outlaiers:
    outliers_mask = (x >= 25) & (x <= 28)
    x[outliers_mask]
    data[outliers_mask] -= 8
    
    # polynomial degree = 2
    A = np.polynomial.polynomial.polyvander(x, deg=2)
    
    matmat = {
        'dumb': matmat_real_dumb,
        'dot': matmat_real_dot,
        'columns': matmat_real_columns,
        'outer': matmat_real_outer,
        'matvec': matmat_real_matvec,
        'numba': matmat_real_numba,
        'complex': matmat_complex
        
    }
    
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function1 not in matmat:
        raise ValueError("Function {} not recognized".format(function1))
        
    if function2 not in matvec:
        raise ValueError("Function {} not recognized".format(function2))
    
    # Weighted least-squares solution
    W = np.ones_like(data)
    W[outliers_mask] = nW
    W = np.diag(W)
    
    # Ordinary least-squares solution
    ATW = matmat[function1](A.T, W)
    ATWA = matmat[function1](ATW, A)
    ATW = matmat[function1](A.T, W)
    ATWd = matvec[function2](ATW, data)
    
    G, y = Gauss_elim(ATWA, ATWd)
    p_WLS = triu_system(G, y)
    
    # Ordinary least-squares solution (final equation)
    y_WLS = p_WLS[0] + p_WLS[1]*x + p_WLS[2]*x*x
    
    return p_WLS, y_WLS


# Mínimos quadrados ponderados iterativamente (IRLS):

def IRLS(x, data, itmax, tol, check_input=True,
                                    function1='dumb', function2='dot'):
    '''
    The least squares will estimate the road parameters
    obs: with iteration
    
    input:
    -> x = 1D array
    -> data = 1D array (data before relation data = ax^2 + bx + c)
    -> max iteration (itmax)
    -> tolerance (tol)
    
    operations:
    ->create outliers
    ->create poli-degree 2
    ->create weighted matrix (W)
    ->create system (A⊤WA)p=A⊤Wd
    ->create y estimated
    
    output:
    return parameters and y estimated
    '''
    p_LS, y_LS = Ordinary_least_squares_solution(x, data)
    
    xmax = np.max(x)
    xmin = np.min(x)
    Dx = xmax - xmin

    dmin = np.min(data)
    dmax = np.max(data)
    Dd = dmax - dmin
    
    # outlaiers:
    outliers_mask = (x >= 25) & (x <= 28)
    x[outliers_mask]
    data[outliers_mask] -= 8
    
    matmat = {
        'dumb': matmat_real_dumb,
        'dot': matmat_real_dot,
        'columns': matmat_real_columns,
        'outer': matmat_real_outer,
        'matvec': matmat_real_matvec,
        'numba': matmat_real_numba,
        'complex': matmat_complex
        
    }
    
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function1 not in matmat:
        raise ValueError("Function {} not recognized".format(function1))
        
    if function2 not in matvec:
        raise ValueError("Function {} not recognized".format(function2))
        
    # polynomial degree = 2
    A = np.polynomial.polynomial.polyvander(x, deg=2)
   
    # IRLS
    
    p0 = p_LS.copy()
    p_IRLS = np.empty_like(p0)
    for i in range(itmax):
        r0 = data - matvec[function2](A, p0)
        R = np.diag(1/np.abs(r0))
        R[r0 < 1e-10] = 1e-10

        ATR = matmat[function1](A.T, R)
        ATRA = matmat[function1](ATR, A)

        ATRd = matvec[function2](ATR, data)
        G, y = Gauss_elim(ATRA, ATRd)
        p_IRLS = triu_system(G, y)
    
        # convergence
        norm_diff = np.linalg.norm(p_IRLS-p0)
        norm_p_IRLS = np.linalg.norm(p_IRLS)
        convergence = norm_diff/(1 + norm_p_IRLS)
        if convergence < tol:
            break
        else:
            p0 = p_IRLS.copy()
            
    y_IRLS = p_IRLS[0] + p_IRLS[1]*x + p_IRLS[2]*x*x
    
    return p_IRLS, y_IRLS


############################################# Steepest decent with exact Line Search ##########################################

##### Obs: adaptado com as funções já implementadas!

def sd_lsearch(A, dobs, p0, tol, itmax, function1='dot', function2='numpy'):
    '''
    Solve a positive-definite linear system by using the 
    method of steepest decent with exact line seach.
    
    Parameters:
    -----------
    A : array 2D
        Symmetric positive definite N x N matrix.
    dobs : array 1D
        Observed data vector with N elements.
    p0 : array 1D
        Initial approximation of the solution p.
    tol : float
        Positive scalar controlling the termination criterion.
    
    Returns:
    --------
    p : array 1D
        Solution of the linear system.
    dpred : array 1D
        Predicted data vector produced by p.
    residuals_L2_norm_values : list
        L2 norm of the residuals along the iterations.
    '''

    A = np.asarray(A)
    dobs = np.asarray(dobs)
    p0 = np.asarray(p0)
    assert A.shape[0] == A.shape[1], 'A must be square'
    assert dobs.size == A.shape[0] == p0.size, 'A order, dobs size and p0 size must be the same'
    assert np.isscalar(tol) & (tol > 0.), 'tol must be a positive scalar'
    assert isinstance(itmax, int) & (itmax > 0), 'itmax must be a positive integer'

    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function1 not in matvec:
        raise ValueError("Function {} not recognized".format(function1))
        
    dot_vec = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'numba': dot_real_numba,
        'complex': dot_complex
    }
    
    if function2 not in dot_vec:
        raise ValueError("Function {} not recognized".format(function2))
        
    N = dobs.size
    p = p0.copy()
    dpred = matvec[function1](A, p)
    # gradient is the residuals vector
    grad = dpred - dobs
    # Euclidean norm of the residuals
    residuals_norm_values = []
    for iteration in range(itmax):
        A_grad = matvec[function1](A, grad)
        grad_A_grad = dot_vec[function2](grad, A_grad)
        mu = dot_vec[function2](grad,grad)/grad_A_grad
        p -= mu*grad
        dpred = matvec[function1](A, p)
        grad = dpred - dobs
        residuals_norm = np.linalg.norm(grad)
        residuals_norm_values.append(residuals_norm)
        if residuals_norm < tol:
            break
    return p, dpred, residuals_norm_values


################################################# Conjugate Gradient Method ##############################################

##### Obs: adaptado com as funções já implementadas!

def cg_method(A, dobs, p0, tol, function1='dot', function2='numpy'):
    '''
    Solve a positive-definite linear system by using the 
    conjugate gradient method (Golub and Van Loan, 2013,
    modified Algorithm 11.3.3, p. 635).
    
    Parameters:
    -----------
    A : array 2D
        Symmetric positive definite N x N matrix.
    dobs : array 1D
        Observed data vector with N elements.
    p0 : array 1D
        Initial approximation of the solution p.
    tol : float
        Positive scalar controlling the termination criterion.
    
    Returns:
    --------
    p : array 1D
        Solution of the linear system.
    dpred : array 1D
        Predicted data vector produced by p.
    residuals_L2_norm_values : list
        L2 norm of the residuals along the iterations.
    '''

    A = np.asarray(A)
    dobs = np.asarray(dobs)
    p0 = np.asarray(p0)
    assert A.shape[0] == A.shape[1], 'A must be square'
    assert dobs.size == A.shape[0] == p0.size, 'A order, dobs size and p0 size must be the same'
    assert np.isscalar(tol) & (tol > 0.), 'tol must be a positive scalar'

    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function1 not in matvec:
        raise ValueError("Function {} not recognized".format(function1))
        
    dot_vec = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'numba': dot_real_numba,
        'complex': dot_complex
    }
    
    if function2 not in dot_vec:
        raise ValueError("Function {} not recognized".format(function2))
        
        
    N = dobs.size
    p = p0.copy()
    # residuals vector
    res = dobs - matvec[function1](A, p)
    # residuals L2 norm
    res_L2 = dot_vec[function2](res,res)
    # Euclidean norm of the residuals
    res_norm = np.sqrt(res_L2)
    # List of Euclidean norm of the residuals
    residuals_norm_values = [res_norm]
    # positive scalar controlling convergence
    delta = tol*np.linalg.norm(dobs)

    # iteration 1
    if res_norm > delta:
        q = res
        w = matvec[function1](A, q)
        mu = res_L2/dot_vec[function2](q,w)
        p += mu*q
        res -= mu*w
        res_L2_ = res_L2
        res_L2 = dot_vec[function2](res,res)
        res_norm = np.sqrt(res_L2)
    
    residuals_norm_values.append(res_norm)
    
    # remaining iterations
    while res_norm > delta:
        tau = res_L2/res_L2_
        q = res + tau*q
        w = matvec[function1](A, q)
        mu = res_L2/dot_vec[function2](q,w)
        p += mu*q
        res -= mu*w
        res_L2_ = res_L2
        res_L2 = dot_vec[function2](res,res)
        res_norm = np.sqrt(res_L2)
        residuals_norm_values.append(res_norm)
    
    dpred = matvec[function1](A,p)

    return p, dpred, residuals_norm_values


# Gradiente conjugado normalizado:

def cgnr_method(A, dobs, p0, tol,  function1='dot', function2='numpy'):
    '''
    Solve a linear system by using the conjugate gradient 
    normal equation residual method (Golub and Van Loan, 2013,
    modified Algorithm 11.3.3 according to Figure 11.3.1 ,
    p. 637).
    
    Parameters:
    -----------
    A : array 2D
        Rectangular N x M matrix.
    dobs : array 1D
        Observed data vector with N elements.
    p0 : array 1D
        Initial approximation of the M x 1 solution p.
    tol : float
        Positive scalar controlling the termination criterion.
    
    Returns:
    --------
    p : array 1D
        Solution of the linear system.
    dpred : array 1D
        Predicted data vector produced by p.
    residuals_L2_norm_values : list
        L2 norm of the residuals along the iterations.
    '''

    A = np.asarray(A)
    dobs = np.asarray(dobs)
    p0 = np.asarray(p0)
    assert dobs.size == A.shape[0], 'A order and dobs size must be the same'
    assert p0.size == A.shape[1], 'A order and p0 size must be the same'
    assert np.isscalar(tol) & (tol > 0.), 'tol must be a positive scalar'

    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function1 not in matvec:
        raise ValueError("Function {} not recognized".format(function1))
        
    dot_vec = {
        'dumb': dot_real_dumb,
        'numpy': dot_real_numpy,
        'numba': dot_real_numba,
        'complex': dot_complex
    }
    
    if function2 not in dot_vec:
        raise ValueError("Function {} not recognized".format(function2))
        
    N = dobs.size
    p = p0.copy()
    # residuals vector
    res = dobs - matvec[function1](A, p)

    # auxiliary variable
    z = matvec[function1](A.T, res) 

    # L2 norm of z
    z_L2 = np.dot(z,z) ################# dot
    # Euclidean norm of the residuals
    res_norm = np.linalg.norm(res)
    # List of Euclidean norm of the residuals
    residuals_norm_values = [res_norm]
    # positive scalar controlling convergence
    delta = tol*np.linalg.norm(dobs)

    # iteration 1
    if res_norm > delta:
        q = z
        w = matvec[function1](A, q) # matvec
        mu = z_L2/dot_vec[function2](w,w) # dot
        p += mu*q
        res -= mu*w
        z = matvec[function1](A.T, res) # matvec
        z_L2_ = z_L2
        z_L2 = dot_vec[function2](z,z) # dot
        res_norm = np.linalg.norm(res)
    
    residuals_norm_values.append(res_norm)
    
    # remaining iterations
    while res_norm > delta:
        tau = z_L2/z_L2_
        q = z + tau*q
        w = np.dot(A, q) # matvec
        mu = z_L2/dot_vec[function2](w,w) # dot
        p += mu*q
        res -= mu*w
        z = matvec[function1](A.T, res) # matvec
        z_L2_ = z_L2
        z_L2 = dot_vec[function2](z,z) # dot
        res_norm = np.linalg.norm(res)
        residuals_norm_values.append(res_norm)
    
    dpred = matvec[function1](A,p)

    return p, dpred, residuals_norm_values


########################################################## Interpolation #################################################

# Lagrange:

def my_lagrange(x, y, xc):
    '''
    Return an interpolated point by applying
    the Lagrange's method.
    
    input
    x: numpy array 1D - x coordinates
    y: numpy array 1D - given values of a function y(x)
    xc: float - coordinate x of the interpolating point
    
    output
    yc: float - interpolated ordinate at xc
    '''
    
    # boolean array
    mask = np.ones(x.size, dtype=bool)
    mask[0] = False
    l = np.empty_like(x)
    for j, xj in enumerate(x):
        l[j] = np.prod(xc - x[mask])
        l[j] /= np.prod(xj - x[mask])
        mask = np.roll(mask,1)
    yc = np.sum(l*y)
    return yc


# Neville 1:

def my_neville(x, y, xc):
    '''
    Return an interpolated point by applying
    the Neville's method.
    
    input
    x: numpy array 1D - x coordinates
    y: numpy array 1D - given values of a function y(x)
    xc: float - coordinate x of the interpolating point
    
    output
    yc: float - interpolated ordinate at xc
    '''
    
    aux = y.copy()
    for j in range(1,x.size):
        for k in range(x.size-j):
            aux[k] = ((xc - x[j+k])*aux[k] + (x[k] - xc)*aux[k+1])/(x[k] - x[j+k])
        
    yc = aux[0]
    return yc


# Neville 2:

def my_neville2(x, y, xc):
    '''
    Return an interpolated point by applying
    the Neville's method.
    
    input
    x: numpy array 1D - x coordinates
    y: numpy array 1D - given values of a function y(x)
    xc: float - coordinate x of the interpolating point
    
    output
    yc: float - interpolated ordinate at xc
    '''
    
    L = x.size
    aux = y.copy()
    for j in range(1,L):
        aux[:L-j] = ((xc - x[j:])*aux[:L-j] + (x[:L-j] - xc)*aux[1:L-j+1])/(x[:L-j] - x[j:])
        
    yc = aux[0]
    return yc


######################################## Spline interpolation with Green's functions - 1D ####################################

# Green 1D:

def G_1D(p, t=0, p_interp=None):
    '''
    Compute matrix G formed by 1D Cartesian Green's functions.
    
    Parameters
    ----------
    p : array 1D
        Vector with N coordinates.

    t : float
        Positive scalar, in the interval [0, 1[ , controlling the
        tension in spline surface.

    p_interp : array 1D
        Vector with N interpolation coordinates.

    Returns
    -------
    G : array 2D
        Matrix of Green's functions.
    '''

    p = np.asarray(p)
    assert p.ndim == 1, 'p must be a vector'
    assert p.size > 2, 'p must have more than two elements'
    assert np.isscalar(t), 't must be a scalar'
    assert (t >= 0) and (t < 1), 't must be greater than or equal to zero and lower lower than one' 

    tau = np.sqrt(t/(1-t))

    if p_interp is not None:
        p_interp = np.asarray(p_interp)
        assert p_interp.ndim == 1, 'p_interp must be a vector'
        assert p_interp.size > 2, 'p_interp must have more than two elements'
        y = p_interp
    else:
        y = p
    
    G = np.empty((y.size, p.size))
    if tau == 0:
        for j, pj in enumerate(p):
            r = np.abs(y - pj)
            G[:,j] = r**3
    else:
        for j, pj in enumerate(p):
            r = np.abs(y - pj)
            G[:,j] = np.exp(-tau*r) + tau*r - 1

    return G


######################################## Spline interpolation with Green's functions - 2D #######################################

# Relevo sintético:

def synthetic_relief(x, y):
    '''
    Compute the synthetic relief model presented by 
    Lancaster and Salkauskas (1980, p. 150).
    
    Parameters
    ----------
    x, y : arrays
        Coordinates x and y of points on the synthetic model.
    
    Returns
    -------
    z : array
        Coordinates z of the model at the given points (x, y).
    '''
    
    shapex = x.shape
    shapey = y.shape
    assert shapex == shapey, 'x and y must have the same shape'

    mask1 = ((y - x) >= 0.5)
    mask2 = ((0 <= (y - x)) & ((y - x) < 0.5))
    mask3 = ((x - 1.5)**2 + (y - 0.5)**2 <= 1/16)

    z = np.zeros_like(x)

    z[mask1] = 1
    z[mask2] = 2*(y[mask2] - x[mask2])
    z[mask3] = 0.5*(np.cos(4*np.pi*np.sqrt((x[mask3] - 1.5)**2 + (y[mask3] - 0.5)**2)) + 1)

    return z


# Green 2D:

def G_2D(p, t=0, p_interp=None):
    '''
    Compute matrix G formed by 2D Cartesian Green's functions.
    
    Parameters
    ----------
    p : array 2D
        Matrix with N rows and 2 columns. The first and second 
        columns contain, respectively, the coordinates x and y
        of the data points.

    t : float
        Positive scalar, in the interval [0, 1[ , controlling the
        tension in spline surface.

    p_interp : array 2D
        Matrix with N rows and 2 columns. The first and second 
        columns contain, respectively, the coordinates x and y
        of the interpolating points.

    Returns
    -------
    G : array 2D
        Matrix of Green's functions.
    '''

    p = np.asarray(p)

    assert p.ndim == 2, 'p must be a matrix'
    assert p.shape[1] == 2, 'p must have two columns'
    assert p.shape[0] > 2, 'p must have more than two rows'
    assert np.isscalar(t), 't must be a scalar'
    assert (t >= 0) and (t < 1), 't must be greater than or equal to zero and lower lower than one' 

    tau = np.sqrt(t/(1-t))

    if p_interp is not None:
        p_interp = np.asarray(p_interp)
        assert p_interp.ndim == 2, 'p_interp must be a matrix'
        assert p_interp.shape[1] == 2, 'p_interp must have two columns'
        assert p_interp.shape[0] > 2, 'p must have more than two rows'
        y = p_interp
    else:
        y = p
    
    G = np.empty((y.shape[0],p.shape[0]))
    if tau == 0:
        for j, (xj, yj) in enumerate(p):
            Dx = y[:,0] - xj
            Dy = y[:,1] - yj
            r = np.sqrt(Dx**2 + Dy**2)
            G[:,j] = r*r*(np.log(r + 1e-15) - 1)
    else:
        for j, (xj, yj) in enumerate(p):
            Dx = y[:,0] - xj
            Dy = y[:,1] - yj
            r = np.sqrt(Dx**2 + Dy**2)
            #G[:,j] = K0(tau*r + 1e-15) + np.log10(tau*r + 1e-15)
            G[:,j] = K0(tau*r + 1e-15) + np.log(tau*r + 1e-15)

    return G


#################################################### 1D Fourier Transform ######################################################

# Geração da matrix F:

def DFT_matrix(N, check_input=True, scale=None, conjugate=False, function='dumb'):
    '''
    The function receive the positive integer N and a string 
    called scale representing the scale factor. 
    This string may assume three possible values: None, 'n' or 'sqrtn'
    
    Obs: define whether or not to use conjugate in "def"
    
    input:
    N = scalar
    
    operation:
    generation FN using a dictionare with outer product functions
    
    output:
    return matrix F
    '''
    
    N = np.asarray(N)
    if check_input is True:
        assert N.ndim == 0, 'x must be a constant'
        assert scale in [None, 'n', 'sqrtn'], "scale must be None, 'n' or 'sqrtn'"
        
    outer = {
        'dumb': outer_real_dumb,
        'numpy': outer_real_numpy,
        'numba': outer_real_numba,
        'complex': outer_complex
    }
    if function not in outer:
        raise ValueError("Function {} not recognized".format(function))
    
    ii = np.linspace(0,N-1,N)
    
    # Geração dos elementos da Matrix H com o produto externo:
    FN = np.exp(-1j*2*np.pi*(outer[function](ii,ii))/N)
    
    # Condicional para escolher o "scale"
    if scale == 'n':
        FN = (1/N)*FN
    if scale == 'sqrtn':
        FN = (1/np.sqrt(N))*FN
    if conjugate == True:
        FN = np.conj(FN)
        
    return FN


# fft:

def fft1D(g, check_input=True, conjugate=True, scale='sqrtn'):
    '''
    The function receive the data vector g and the string scale
    representing the scale factor (the same defined for your function DFT_matrix). 
    The function fft1D return the Discrete Fourier Transform of g. 

    This function create the DFT matrix H by using your function DFT_matrix
    
    Obs: define whether or not to use conjugate in "def"
    
    input:
    g = 1D array
    
    operation:
    Using the function DFT_matrix to compute F
    And using matvec_complex to multiply F and g.
    
    output:
    return G transform
    '''
    
    g = np.asarray(g)
    if check_input is True:
        assert g.ndim == 1, 'g must be a 1D array'
        assert scale in [None, 'n', 'sqrtn'], "scale must be None, 'n' or 'sqrtn'"
        
    M = np.size(g)
    G = np.zeros((M,M))
    
    # Condicional para escolher o "scale"
    if scale == None:
        F = DFT_matrix(M, scale=None)
        G = matvec_complex(F,g)
    if scale == 'n':
        F = DFT_matrix(M, scale='n')
        G = matvec_complex(F,g)
    if scale == 'sqrtn':
        F = DFT_matrix(M, scale='sqrtn')
        G = matvec_complex(F,g)
    if conjugate == True:
        G = np.conj(G)
        
    return G


# ifft:

def ifft1D(gg, check_input=True, conjugate=True, scale='sqrtn', function='dumb'):
    '''
    The function receive a complex vector G and 
    the string scale representing the scale factor. 
    The function ifft1D return the Inverse Discrete Fourier Transform of G
    
    Obs: define whether or not to use conjugate in "def"
    
    input:
    gg = 1D array complex
    
    operation:
    generation HN using a dictionare with outer product functions
    And calculate G complex with matvec_complex function
    
    output:
    return vector G inverse transform
    '''
    
    gg = np.asarray(gg)
    if check_input is True:
        assert gg.ndim == 1, 'gg must be a complex vector'
        assert scale in [None, 'n', 'sqrtn'], "scale must be None, 'n' or 'sqrtn'"
    
    M = np.size(gg)
    G = np.zeros(M)
    
    # Condinional para escolher o "scale"
    if scale == None:
        FN = DFT_matrix(M, scale=None) #, conjugate=False)
        H = (1/M)*np.conj(FN)
        G = matvec_complex(H, gg)
    if scale == 'n':
        FN = DFT_matrix(M, scale='n') #,conjugate=False)
        H = M*np.conj(FN)
        G = matvec_complex(H, gg)
    if scale == 'sqrtn':
        FN = DFT_matrix(M, scale='sqrtn') #,conjugate=False)
        H = np.conj(FN)
        G = matvec_complex(H, gg)
    if conjugate==True:
        G = np.conj(G)
        
    return G
    
    
###################################################### Convolution #########################################################

# Convolução circular com matvec:

def circular_convolution_matvec(a, b, check_input=True, function='dumb'):
    '''
    The function receive the vectors a and b and return the vector w.
    The function compute the convolution by using the matrix-vector.
    
    Obs: This code use Circulation matrix
    
    input:
    a = 1D array
    b = 1D array
    
    operation:
    -> Using "circulatant" from scipy to create: C 2D array
    -> Matvec product (relation: w = C*a)
    
    output:
    return w result of convolution
    '''
    
    N = np.size(a)
    a = np.asarray(a)
    b = np.asarray(b)
    if check_input is True:
        assert a.ndim == 1, 'a must be a 1D array'
        assert b.ndim == 1, 'b must be a 1D array'
        assert np.size(b) == N, 'a and b must be same shape'
        
    matvec = {
        'dumb': matvec_real_dumb,
        'dot': matvec_real_dot,
        'columns': matvec_real_columns,
        'numba': matvec_real_numba,
        'complex': matvec_complex
    }
    
    if function not in matvec:
        raise ValueError("Function {} not recognized".format(function))
        
    C = circulant(b)
    w = np.zeros(N)
    
    w = matvec[function](C, a)
    
    return w
        
    
# Convolução circular com Fourier:

def circular_convolution_dft(a, b, check_input=True):
    '''
    The function receive the vectors a and b and return the vector w.
    The function must compute the convolution by using the Fourier.
    
    input:
    a = 1D array
    b = 1D array
    
    operation:
    -> fft for vectors a and b;
    -> Hadamard produt;
    -> ifft to calculate w
    
    output:
    return w result of convolution
    '''
    
    M = np.size(a)
    a = np.asarray(a)
    b = np.asarray(b)
    if check_input is True:
        assert a.ndim == 1, 'a must be a 1D array'
        assert b.ndim == 1, 'b must be a 1D array'
        assert np.size(b) == M, 'a and b must be same shape'
        
    Ha = fft1D(a, scale='sqrtn', conjugate=False)
    Hb = fft1D(b, scale='sqrtn', conjugate=False)
    
    Hada = (np.sqrt(M)*hadamard_complex(Hb,Ha, function='numpy'))
    
    w = ifft1D(Hada, conjugate=True, scale='sqrtn').real
    
    return w


# Convolução linear com matvec:

def linear_convolution_matvec(a, b, check_input=True, function='dumb'):
    '''
    The function receive the vectors a and b and return the vector w.
    The function compute the convolution by using the matrix-vector
    
    Obs: This code use Toeplitz matrix.
    
    input:
    a = 1D array
    b = 1D array
    
    operation:
    -> Using "toeplitz" from scipy to create: B 2D array
    -> Matvec product (relation: w = B*a)
    
    output:
    return w result of convolution
    '''
    
    N = np.size(a)
    a = np.asarray(a)
    b = np.asarray(b)
    if check_input is True:
        assert a.ndim == 1, 'a must be a 1D array'
        assert b.ndim == 1, 'b must be a 1D array'
        
    Na = np.size(a)
    Nb = np.size(b)
    
    Nw = Na + Nb - 1
    N = Na + Nb
    
    a_padd = np.hstack([a, np.zeros(Nb)])

    b_padd = np.hstack([b, np.zeros(Na)])
    
    T = toeplitz(b_padd, np.zeros(N))
    
    w = matvec_complex(T, a_padd)[:-1]
    
    return w
        
    
# Convolução linear com Fourier:

def linear_convolution_dft(a, b, check_input=True):
    '''
    The function receive the vectors a and b and return the vector w.
    The function compute the convolution by using the Fourier.
    
    input:
    a = 1D array
    b = 1D array
    
    operation:
    -> fft for vectors a_padd and b_padd;
    -> Hadamard produt;
    -> ifft to calculate w
    
    output:
    return w result of convolution
    '''
    
    N = np.size(a)
    a = np.asarray(a)
    b = np.asarray(b)
    if check_input is True:
        assert a.ndim == 1, 'a must be a 1D array'
        assert b.ndim == 1, 'b must be a 1D array'
        
    Na = np.size(a)
    Nb = np.size(b)
    
    Nw = Na + Nb - 1
    N = Na + Nb
    
    a_padd = np.hstack([a, np.zeros(Nb)])

    b_padd = np.hstack([b, np.zeros(Na)])
    
    Ha_padd = fft1D(a_padd, scale='sqrtn', conjugate=False)
    Hb_padd = fft1D(b_padd, scale='sqrtn', conjugate=False)
    
    Hada_padd = (np.sqrt(N)*hadamard_complex(Hb_padd,Ha_padd, function='numpy'))
    
    w = ifft1D(Hada_padd, conjugate=True, scale='sqrtn').real[:-1]
    
    
    return w
        

################################################## Correlation (conv cont.)#####################################################
    
#correlação com matvec:

def correlation_matvec(a, b, check_input=True):
    '''
    Compute the crosscorrelation of the 1D arrays. 
    Uses the DFT of the arrays to compute the crosscorrelation.
    
    input:
    a = 1D array
    b = 1D array
    
    operation:
    -> creation variables a_padd and b_padd
    -> Toeplitz matrix creation
    -> matvec product
    
    output:
    
    Returns: crosscorrelation of x and y

    '''
    
    N = np.size(a)
    M = np.size(b)
    a = np.asarray(a)
    b = np.asarray(b)
    if check_input is True:
        assert a.ndim == 1, 'a must be a 1D array'
        assert b.ndim == 1, 'b must be a 1D array'

    a_padd = np.hstack([a, np.zeros(M)])
    b_padd = np.hstack([np.conjugate(b[::-1]), np.zeros(N)])

    T = toeplitz(b_padd, np.zeros(M + N))
    w_cross_matvec_Ta = matvec_complex(T, a_padd)[:-1]

    return w_cross_matvec_Ta


# Correlação com Fourier:

def correlation_dft(a, b, check_input=True):
    '''
    Compute the crosscorrelation of the 1D arrays. 
    Uses the DFT of the arraysto compute the crosscorrelation.

    input:
    a = 1D array
    b = 1D array
    
    operation:
    -> creation variables a_padd and b_padd
    -> Hadamard product
    -> ifft
    
    output:
    
    Returns: crosscorrelation of x and y
    
    '''
    
    N = np.size(a)
    M = np.size(b)
    a = np.asarray(a)
    b = np.asarray(b)
    if check_input is True:
        assert a.ndim == 1, 'a must be a 1D array'
        assert b.ndim == 1, 'b must be a 1D array'

    a_padd = np.hstack([a, np.zeros(M)])
    b_padd = np.hstack([np.conjugate(b[::-1]), np.zeros(N)])
    
    Ha = fft1D(a_padd, scale='sqrtn', conjugate=False)
    Hb = fft1D(b_padd, scale='sqrtn', conjugate=False)
    
    Hada = (np.sqrt(M+N)*hadamard_complex(Hb,Ha, function='numpy'))

    w = ifft1D(Hada, conjugate=True, scale='sqrtn').real[:-1]

    return w



############################################################### FIM ############################################################
