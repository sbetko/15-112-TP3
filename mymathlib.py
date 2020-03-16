# This file houses all of the required mathematical functions and
# matrix/vector operations used in the NeuralNetworkApp.

import math, random, copy, numbers
import numpy as np

# with support for up to 2 dimensional lists
def almostEqual(a, b):
    if type(a) == type(b) == list and len(a) == len(b):
        listIs2d = False
        for item in a:
            if isinstance(item, list):
                listIs2d = True
        for i in range(len(a)):
            if listIs2d:
                for j in range(len(a[0])):
                    if abs(a[i][j] - b[i][j]) > 10**-9:
                        return False
            if not listIs2d:
                if abs(a[i] - b[i]) > 10**-9:
                    return False
        return True
    else:
        return abs(a - b) <= 10**-9

# Returns true if the specified (x,y) pair exists within the circle of radius r
def pointInCircle(r, circleCenter, point):
    circleCenterX, circleCenterY = circleCenter
    x1, y1 = point
    return (circleCenterX - r < x1 < circleCenterX + r and
            circleCenterY - r < y1 < circleCenterY + r)

# Returns true if the point is in the rectangular bounds specified.
# Does not assume sorted bounds
def pointInBounds(point, bounds):
    ax1, ay1, ax2, ay2 = bounds
    ax1, ax2 = sorted([ax1, ax2])
    ay1, ay2 = sorted([ay1, ay2])
    x1, y1 = point
    return ax1 < x1 < ax2 and ay1 < y1 < ay2

# Returns the column of the 2d list as a 1d list
def getColumn(M, col):
    return [row[col] for row in M]

# Bounds a value between an upper and a lower limit
def getInBounds(x, lower, upper):
    if x < lower:
        return lower
    elif x > upper:
        return upper
    else:
        return x

# Karlik, Bekir, and A. Vehbi Olgac. "Performance analysis of various activation
# functions in generalized MLP architectures of neural networks." International
# Journal of Artificial Intelligence and Expert Systems 1.4 (2011): 111-122.

# The hyperbolic tangent function, may take a 2d, 1d, or scalar input
# where order is the order of the derivative, and order 0 is the 0th order
# derivative
def tanH(x, order = 0):
    if order == 0:
        return np.tanh(x)
    elif order == 1:
        return 1 - np.tanh(x)**2
    '''
    if isinstance(x, numbers.Number):
        if order == 0:
            e = math.e
            return (e**(2*x) - 1)/(e**(2*x) + 1)
        elif order == 1:
            return 1 - tanH(x)**2
    elif type(x) == list:
        return applyFunctionToMatrix(tanH, x, order)
        '''

# The logistic function, may take a 2d, 1d, or scalar input
# where order is the order of the derivative, and order 0 is the 0th order
# derivative
def logistic(x, order = 0):
    z = 1 / (1 + np.exp(-x))
    if order == 0:
        return z
    elif order == 1:
        return z * (1 - z)

# Applies the provided function to a 2d or 1d list
# where order is the order of the derivative, and order 0 is the 0th order
# derivative
def applyFunctionToMatrix(f, M, order):
    rows, cols = len(M), len(M[0])
    res = make2dList(rows, cols)
    for i in range(rows):
        for j in range(cols):
            res[i][j] = f(M[i][j], order = order)
    return res

# Returns the dot product of two vectors
def dotProduct(a, b):
    if isinstance(a, np.ndarray):
        return np.dot(a, b, out = None)
    else:
        if len(a) != len(b):
            return None
        res = 0
        for i in range(len(a)):
            res += (a[i][0]*b[i][0])
        return res

# Returns the matrix product of two matrices (M x N)
def matProd(M, N):
    if isinstance(M, np.ndarray):
        return np.matmul(M, N)
    else:
        try:
            res = make2dList(len(M), len(N[0]))
        except:
            return
        for mRow in range(len(M)):
            for nCol in range(len(N[0])):
                for nRow in range(len(N)):
                    res[mRow][nCol] += M[mRow][nRow] * N[nRow][nCol]
        return res

# Adds two vectors component-wise
def addVectors(a, b, sign = 1):
    return a + sign*b

    '''
    if len(a) != len(b):
        return None
    res = []
    for i in range(len(a)):
        aElem = a[i] if isinstance(a[i], numbers.Number) else a[i][0]
        bElem = b[i] if isinstance(b[i], numbers.Number) else b[i][0]
        res.append(aElem + sign*bElem)
    return transpose(res)
    '''

# Transposes the given matrix
def transpose(M):
    return np.transpose(M)
    
    '''
    # If there are no inner lists, then rows of that "2d list" = 1
    hasRows = False
    for elem in M:
        if type(elem) == list:
            hasRows = True

    cols = len(M) if hasRows else 1
    rows = len(M[0]) if hasRows else len(M)
    res = make2dList(rows, cols)
    for i in range(cols):
        for j in range(rows):
            if hasRows:
                res[j][i] = M[i][j]
            else:
                res[j][i] = M[j]
    return res
    '''

# Performs MSE for a single training example
# where order is the order of the derivative, and order 0 is the 0th order
# derivative
def MSE(observed, actual, order = 0):
    observed = observed.flatten()
    actual = actual.flatten()
    if order == 0:
        # perform MSE over entire vector
        diff = (addVectors(observed, actual, -1))
        return 1/2 * dotProduct(diff, diff)
    else:
        # perform elementwise MSE prime
        return addVectors(observed, actual, -1)

# Elementwise matrix product
def hadamardProd(A, B):
    return np.array(A) * np.array(B)

# Performs elementwise multiplication on a matrix by a scalar
def multiplyMatrixByScalar(n, M):
    if isinstance(M, np.ndarray):
        return n * M
    else:
        res = M[:]
        for i in range(len(M)):
            for j in range(len(M[0])):
                res[i][j] = M[i][j] * n
        return res

# Returns the elementwise sum A + B
def matrixSum(A, B, sign = 1):
    if isinstance(A, np.ndarray):
        return A + sign*B
    else:
        res = []
        for i in range(len(A)):
            res.append([])
            for j in range(len(A[0])):
                res[i].append(A[i][j] + sign*B[i][j])
        return res

# Returns an empty 2d list with the number of rows and columns specified
def make2dList(rows, cols):
    return np.zeros((rows, cols))
    '''
    return [[0]*cols for row in range(rows)]
    '''

# Returns a 2d list with each element sampled from the normal distribution
def makeGaussian2dList(rows, cols, mu, sigma):
    return np.random.randn(rows, cols)
    '''
    L = make2dList(rows, cols)
    for i in range(rows):
        for j in range(cols):
            L[i][j] = random.gauss(mu, sigma)
    return L
    '''

# Flattens a 2d list into a 1d list
def flatten2dList(lst):
    '''
    res = []
    for row in range(len(lst)):
        for col in range(len(lst[row])):
            res.append(lst[row][col])
    return res
    '''
    return [y for x in lst for y in x]

# Tests the functions in this library
def testMathHelpers():
    # Test transpose
    M = [[1, 1, 1, 6], [0, 2, -1, 3], [4, 0, 10, 42]]
    actual = [[1,0,4], [1,2,0], [1,-1,10], [6,3,42]]
    observed = transpose(M)
    assert(actual == observed)

    v = [1,2,3,4]
    actual = [[1],[2],[3],[4]]
    observed = transpose(v)
    assert(actual == observed)

    # Test matProd
    M = [[1, 2],
         [3, 4],
         [5, 6]]
    x = [[0],
         [1]]
    actual = [[0+2*1], [0+1*4], [0+6*1]]
    observed = matProd(M, x)
    assert(observed == actual)

    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    B = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    
    # Test hadamardProd
    actual = [[1,  4,  9 ],
              [16, 25, 36],
              [49, 64, 81]]
    observed = hadamardProd(A, B)
    assert(observed == actual)

    # Test matrixSum
    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    B = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    
    actual = [[2, 4, 6],
              [8, 10, 12],
              [14, 16, 18]]
    observed = matrixSum(A, B)
    assert(observed == actual)

    # Test MSE
    a = [[20]]
    y = [[1]]
    
    actual = [[19]]
    observed = MSE(a, y, order = 1)
    assert(observed == actual)

    # Test addVectors
    a = [[1],[2],[3],[4]]
    b = [[2],[3],[4],[5]]
    actual = [[3],[5],[7],[9]]
    observed = addVectors(a, b)
    assert(observed == actual) 