#ifndef _MATSOFVECS_H_
#define _MATSOFVECS_H_
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include "floatcast.h"

namespace vltools {

template <typename T>
void sumsOfSubvectorComponents (
	std::vector<typename FloatCast<T>::Type> & sums,
	const std::vector<T> & data,
	const uint64_t & subvectorLength)
{
	if (data.size() % subvectorLength != 0) {
		std::cout << "sumsOfSubvectorComponents: ";
		std::cout << "invalid data size." << std::endl;
		return;
	}
	std::size_t numSubvectors = data.size() / subvectorLength;
	sums.resize(numSubvectors);
	const T * v = &data[0];
	for (std::size_t i = 0; i < numSubvectors; i++) {
		sums[i] = 0.0;
		for (std::size_t j = 0; j < subvectorLength; j++)
			sums[i] += *v++;
	}
}

template <typename T>
void sumsOfSquaresOfSubvectorComponents (
	std::vector<typename FloatCast<T>::Type> & sumsOfSquares,
	const std::vector<T> & data,
	const uint64_t & subvectorLength)
{
	if (data.size() % subvectorLength != 0) {
		std::cout << "sumsOfSquaresOfSubvectorComponents: ";
		std::cout << "invalid data size." << std::endl;
		return;
	}
	std::size_t numSubvectors = data.size() / subvectorLength;
	sumsOfSquares.resize(numSubvectors);
	const T * v = &data[0];
	for (std::size_t i = 0; i < numSubvectors; i++) {
		sumsOfSquares[i] = 0.0;
		for (std::size_t j = 0; j < subvectorLength; j++)
			sumsOfSquares[i] += v++[0]*v[0];
	}
}

template <typename T>
void identityTransformationConstants (
	std::vector<typename FloatCast<T>::Type> & a,
	std::vector<typename FloatCast<T>::Type> & b,
	std::vector<bool> & mask,
	const std::vector<T> & data,
	const std::vector<uint64_t> & M,
	const std::vector<uint64_t> & N,
	const uint64_t & subvectorLength,
	const uint64_t & m,
	const uint64_t & n)
{
	typedef typename FloatCast<T>::Type FloatType;

	if (data.size() % subvectorLength != 0) {
		std::cout << "identityTransformationConstants: ";
		std::cout << "invalid data size." << std::endl;
        return;
	}

	// simply set a to 1's and b to 0's
	std::size_t numSubvectors = data.size() / subvectorLength;
	a.resize(numSubvectors, 1);
	b.resize(numSubvectors, 0);

	// set mask
	mask.resize(numSubvectors, false);
	std::size_t numLevels = M.size();
	std::size_t offset = 0;
	for (std::size_t l = 0; l < numLevels; l++) {
        if (m <= M[l] && n <= N[l]) {
            for (std::size_t j = 0; j < N[l] - n + 1; j++)
                for (std::size_t i = 0; i < M[l] - m + 1; i++)
                    mask[offset + j * M[l] + i] = true;
        }
		offset += M[l] * N[l];
	}
}

template <typename T>
void unitLengthTransformationConstants (
	std::vector<typename FloatCast<T>::Type> & a,
	std::vector<typename FloatCast<T>::Type> & b,
	std::vector<bool> & mask,
	const std::vector<T> & data,
	const std::vector<uint64_t> & M,
	const std::vector<uint64_t> & N,
	const uint64_t & subvectorLength,
	const uint64_t & m,
	const uint64_t & n)
{
	typedef typename FloatCast<T>::Type FloatType;

	if (data.size() % subvectorLength != 0) {
		std::cout << "unitLengthTransformationConstants: ";
		std::cout << "invalid data size." << std::endl;
		return;
	}

	// compute sums and sums of squares of subvector components
	std::vector<FloatType> subVec_sumsOfSquares;
	sumsOfSquaresOfSubvectorComponents(subVec_sumsOfSquares, data, subvectorLength);
	std::size_t numSubvectors = subVec_sumsOfSquares.size();

	// compute sums and sums of squares of vector components
	std::vector<FloatType> vec_sumsOfSquares(numSubvectors, 0.0);
	std::size_t numLevels = M.size();
	std::size_t offset = 0;
	for (std::size_t l = 0; l < numLevels; l++) {
		boxFilter2D(&vec_sumsOfSquares[offset], &subVec_sumsOfSquares[offset], M[l], N[l], m, n);
		offset += M[l] * N[l];
	}

	// compute a and b
    mask.resize(numSubvectors);
	a.resize(numSubvectors);
	b.resize(numSubvectors, 0.0);
	for (std::size_t i = 0; i < numSubvectors; i++) {
        if (!(mask[i] = vec_sumsOfSquares[i] != 0.0)) continue;
		FloatType norm = sqrt(vec_sumsOfSquares[i]);
		a[i] = (FloatType)1.0 / norm;
		mask[i] = norm != 0.0;
	}

}

template <typename T>
void zeroMeanUnitLengthTransformationConstants (
	std::vector<typename FloatCast<T>::Type> & a,
	std::vector<typename FloatCast<T>::Type> & b,
	std::vector<bool> & mask,
	const std::vector<T> & data,
	const std::vector<uint64_t> & M,
	const std::vector<uint64_t> & N,
	const uint64_t & subvectorLength,
	const uint64_t & m,
	const uint64_t & n)
{
	typedef typename FloatCast<T>::Type FloatType;

	if (data.size() % subvectorLength != 0) {
		std::cout << "zeroMeanUnitLengthTransformationConstants: ";
		std::cout << "invalid data size." << std::endl;
		return;
	}

	// compute sums and sums of squares of subvector components
	std::vector<FloatType> subVec_sums;
	std::vector<FloatType> subVec_sumsOfSquares;
	sumsOfSubvectorComponents(subVec_sums, data, subvectorLength);
	sumsOfSquaresOfSubvectorComponents(subVec_sumsOfSquares, data, subvectorLength);
	std::size_t numSubvectors = subVec_sums.size();

	// compute sums and sums of squares of vector components
	std::vector<FloatType> vec_sums(numSubvectors, 0.0);
	std::vector<FloatType> vec_sumsOfSquares(numSubvectors, 0.0);
	std::size_t numLevels = M.size();
	std::size_t offset = 0;
	for (std::size_t l = 0; l < numLevels; l++) {
		boxFilter2D(&vec_sums[offset], &subVec_sums[offset], M[l], N[l], m, n);
		boxFilter2D(&vec_sumsOfSquares[offset], &subVec_sumsOfSquares[offset], M[l], N[l], m, n);
		offset += M[l] * N[l];
	}

	// compute a and b
	std::size_t vectorLength = m * n * subvectorLength;
    mask.resize(numSubvectors);
	a.resize(numSubvectors);
	b.resize(numSubvectors);
	for (std::size_t i = 0; i < numSubvectors; i++) {
        if (!(mask[i] = vec_sumsOfSquares[i] != 0.0)) continue;
		FloatType c = sqrt(vec_sumsOfSquares[i] - vec_sums[i] * vec_sums[i] / vectorLength);
		a[i] = (FloatType)1.0 / c;
		b[i] = -vec_sums[i] / c / vectorLength;
	}
}

template <typename T>
void boxFilter2D (
	typename FloatCast<T>::Type * out,
	const T * in,
	const std::size_t & M,
	const std::size_t & N,
	const std::size_t & m,
	const std::size_t & n)
{
	typedef typename FloatCast<T>::Type FloatType;
    if (m > M || n > N) return;
	std::size_t Mres = M - m + 1;
	std::size_t Nres = N - n + 1;
	const T * x = in;	// input
	FloatType * y = out;	// output
	for (std::size_t j = 0; j < Nres; j++) {
		for (std::size_t i = 0; i < Mres; i++) {
			*y = 0.0;
			const T * v = x;
			for (std::size_t jj = 0; jj < n; jj++) {
				for (std::size_t ii = 0; ii < m; ii++) {
					*y += *v++;
				}
				v += (M - m);
			}
			x++;
			y++;
		}
		x += (m - 1);
		y += (m - 1);
	}
}

namespace MatOfVec {
	template <typename U> struct TypeID {static const uint8_t index = 255;};
	template <> struct TypeID<float> {static const uint8_t index = 0;};
	template <> struct TypeID<double> {static const uint8_t index = 1;};
	template <> struct TypeID<uint8_t> {static const uint8_t index = 2;};
	template <> struct TypeID<uint16_t> {static const uint8_t index = 3;};
	template <> struct TypeID<uint32_t> {static const uint8_t index = 4;};
	template <> struct TypeID<uint64_t> {static const uint8_t index = 5;};
	template <> struct TypeID<int8_t> {static const uint8_t index = 6;};
	template <> struct TypeID<int16_t> {static const uint8_t index = 7;};
	template <> struct TypeID<int32_t> {static const uint8_t index = 8;};
	template <> struct TypeID<int64_t> {static const uint8_t index = 9;};
}

template <typename T>
void readMatricesOfVectors (
	std::vector<T> & data,
	std::vector<uint64_t> & M,
	std::vector<uint64_t> & N,
	uint64_t & subvectorLength,
	const char * filename)
{
	std::ifstream in(filename, std::ios_base::binary | std::ios_base::in);
	if (!in.is_open()) {
		std::cout << "readMatricesOfVectors: ";
		std::cout << "failed to open file." << std::endl;
		return;
	}

	// Calculate file length
	in.seekg(0,in.end);
	std::size_t fileSize = in.tellg();
	in.seekg(0,in.beg);

	// First byte: element type

	uint8_t elementType;
	in.read((char*)&elementType, 1);

	if (elementType != MatOfVec::TypeID<T>::index) {
		std::cout << "readMatricesOfVectors: ";
		std::cout << "expecting element type " << MatOfVec::TypeID<T>::index << ", ";
		std::cout << "instead found element type " << elementType << "." << std::endl;
		return;
	}

	// Next 8 bytes: subvector length
	in.read((char*)&subvectorLength, 8);
	if (in.bad()) {
		std::cout << "readMatricesOfVectors: ";
		std::cout << "failed reading subvector length." << std::endl;
		return;
	}

	// Next 8 bytes: number of levels
	uint64_t numLevels;
	in.read((char*)&numLevels, 8);
	if (in.bad()) {
		std::cout << "readMatricesOfVectors: ";
		std::cout << "failed reading number of levels." << std::endl;
		return;
	}

	// Next 8 bytes x number of levels: # rows for each level
	M.resize(numLevels);
	in.read((char*)&M[0], 8 * numLevels);
	if (in.bad()) {
		std::cout << "readMatricesOfVectors: ";
		std::cout << "failed reading rows per level." << std::endl;
		return;
	}

	// Next 8 bytes x number of levels: # cols for each level
	N.resize(numLevels);
	in.read((char*)&N[0], 8 * numLevels);
	if (in.bad()) {
		std::cout << "readMatricesOfVectors: ";
		std::cout << "failed reading cols per level." << std::endl;
		return;
	}

	// Compute size of data
	std::size_t numCells = 0;
	for (std::size_t i = 0; i < numLevels; i++) {
		numCells += M[i] * N[i];
	}
	std::size_t numData = numCells * subvectorLength;

	// Check file size
	std::size_t expectedFileSize = 
		1 + 8 + 8 + 2 * numLevels * 8 + numData * sizeof(T);
	if (fileSize != expectedFileSize) {
		std::cout << "readMatricesOfVectors: ";
		std::cout << "actual file size (" << fileSize << ") ";
		std::cout << "different from expected file size ";
		std::cout << "(" << expectedFileSize << ")" << std::endl;
		return;
	}

	// Copy remaining bytes into data
	data.resize(numData * sizeof(T));
	in.read((char*)&data[0], numData * sizeof(T));
	if (in.bad()) {
		std::cout << "readMatricesOfVectors: ";
		std::cout << "failed reading data." << std::endl;
		return;
	}
}

template <typename T>
void concatenateArrays (
	std::vector<T> & concatenated,
	const std::vector<const T*> & arrays,
	const std::vector<std::size_t> & sizes)
{
	// calculate output size
	std::size_t totalSize = 0;
	for (std::size_t i = 0; i < sizes.size(); i++)
		totalSize += sizes[i];
	concatenated.resize(totalSize);

	// copy arrays to output array
	std::size_t offset = 0;
	for (std::size_t i = 0; i < sizes.size(); i++) {
		std::copy(arrays[i], arrays[i] + sizes[i], &concatenated[offset]);
		offset += sizes[i];
	}
}

}	// namespace vltools
#endif
