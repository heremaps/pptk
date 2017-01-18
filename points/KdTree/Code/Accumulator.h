#ifndef __ACCUMULATOR_H__
#define __ACCUMULATOR_H__

namespace pointkd {
	// specify how element types are converted to floating point number
	template <typename T> struct Accumulator {};
	template <> struct Accumulator<unsigned char> {
		typedef float Type; };
	template <> struct Accumulator<unsigned int> {
		typedef float Type; };
	template <> struct Accumulator<char> {
		typedef float Type; };
	template <> struct Accumulator<int> {
		typedef float Type; };
	template <> struct Accumulator<float> {
		typedef float Type; };
	template <> struct Accumulator<double> {
		typedef double Type; };
}
#endif
