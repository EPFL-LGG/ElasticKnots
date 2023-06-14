#ifndef SPARSE_MATRICES_UTILITIES_HH
#define SPARSE_MATRICES_UTILITIES_HH

#include <MeshFEM/SparseMatrices.hh>

// Extend `result` from size (n x n) to (finalSize x finalSize) by adding empty columns/rows on the right/bottom
inline SuiteSparseMatrix extendSparseMatrixSouthEast(SuiteSparseMatrix &result, size_t finalSize) {
    assert(result.m == result.n && result.n <= (long)finalSize);
    const size_t additionalColumns = finalSize - result.n;
    result.m = result.n = finalSize;
    result.Ap.reserve(finalSize + 1);
    for (size_t i = 0; i < additionalColumns; i++)
        result.Ap.push_back(result.nz);
    return result;
}

#endif /* end of include guard: SPARSE_MATRICES_UTILITIES_HH */