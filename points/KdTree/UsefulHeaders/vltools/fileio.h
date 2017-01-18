/* fileio.h
   */
#ifndef __FILEIO_H__
#define __FILEIO_H__
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
namespace vltools {

template <typename T>
void readpoints (
		std::vector<T> & data,
		const char * filename,
		unsigned int dim)
{
	std::ifstream infile(filename, std::ios_base::binary | std::ios_base::in);
	if (infile.fail()) {
		std::cerr << "readpoints: failed to open " << filename << std::endl;
		exit(1);
	}

	// compute number of points
	infile.seekg(0, std::ios::end);
	size_t length = (size_t)infile.tellg();
	infile.seekg(0, std::ios::beg);
	size_t numval = (size_t)length / sizeof(T);

	if (numval % dim != 0) {
		std::cerr << "readpoints: invalid .points file" << std::endl;
		exit(1);
	}

	data.clear();
	data.resize(numval);
	infile.read((char*)&data[0], length);

	infile.close();
}

template <typename T>
void writepoints (
		const char * filename,
		const std::vector<T> & data, 
		unsigned int dim)
{
	if (data.size() == 0) {
		std::cerr << "writepoints: nothing to write... " << std::endl;
		return;
	}

	if (data.size() % dim != 0) {
		std::cerr << "writepoints: invalid input data and dim" << std::endl;
		exit(1);
	}

	std::ofstream outfile(filename, std::ios::binary | std::ios::out);
	if (outfile.fail()) {
		std::cerr << "writepoints: failed to open " << filename << ", exiting..." << std::endl;
		exit(1);
	}

	size_t numBytes = (size_t)data.size() * sizeof(T);
	outfile.write((char*)&data[0], numBytes);
	std::cerr << "writepoints: wrote " << data.size() / dim << " " << 
		dim << "-dimensional points to " << filename << std::endl;
	outfile.close();
}

/**
* @brief Writes out dim-dimensional points from the 'data' vector into
* a binary file.  'usePoints' is a boolean mask that indicates when a
* point should be written to the file.
*
* This function was written to facilitate _fast_ writing of data where
* we only wanted to select a subset of the data to write without having
* to create duplicates (hence wasting memory).
*
* @param filename Filename of binary file to be written to.
* @param data Vector of type T which holds the dim-dimensional packed
* points.
* @param usePoints A type of mask where usePoints[i] is a boolean that
* indicates if point i in 'data' should be written to the binary file.
* i is not an index that can be directly used in 'data' unless the
* points are 1D.
* @param dim Dimensions per point.
*/
template <typename T>
void writepoints (
		const char * filename,
		const std::vector<T> & data,
		const std::vector<bool> &usePoints,
		unsigned int dim)
{
	if (data.size() == 0) {
		std::cerr << "writepoints: nothing to write... " << std::endl;
		return;
	}

	if (data.size() % dim != 0) {
		std::cerr << "writepoints: invalid input data and dim" << std::endl;
		exit(1);
	}

	std::ofstream outfile(filename, std::ios::binary | std::ios::out);
	if (outfile.fail()) {
		std::cerr << "writepoints: failed to open " << filename << ", exiting..." << std::endl;
		exit(1);
	}

	unsigned int numWritten = 0;

	// Write out each point.
	for (size_t i = 0; i < usePoints.size(); ++i)
	{
		if (usePoints[i])
		{
			// Write the first dimension through the last dimension.
			size_t index = (size_t)i * (size_t)dim;
			outfile.write((char*)&data[index], sizeof(T) * dim);
			++numWritten;
		}
	}

	std::cerr << "writepoints: wrote " << numWritten << " " <<
		dim << "-dimensional points to " << filename << std::endl;

	outfile.close();
}

template <typename T>
void writepoints (
		const char * filename,
		const std::vector<T> & data,
		const std::vector<unsigned int> &indices,
		unsigned int dim)
{
	if (data.size() == 0) {
		std::cerr << "writepoints: nothing to write... " << std::endl;
		return;
	}

	if (data.size() % dim != 0) {
		std::cerr << "writepoints: invalid input data and dim" << std::endl;
		exit(1);
	}

	std::ofstream outfile(filename, std::ios::binary | std::ios::out);
	if (outfile.fail()) {
		std::cerr << "writepoints: failed to open " << filename << ", exiting..." << std::endl;
		exit(1);
	}

	// Write out each point.
	for (unsigned int i = 0; i < indices.size(); ++i)
	{
		outfile.write((char *)&data[indices[i]], sizeof(T) * dim);
	}

	std::cerr << "writepoints: wrote " << indices.size() << " " <<
		dim << "-dimensional points to " << filename << std::endl;

	outfile.close();
}

} // namespace vltools
#endif
