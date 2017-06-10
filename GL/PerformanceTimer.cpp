/*
 * The MIT License
 *
 * Copyright (c) 2010 Paul Solt, PaulSolt@gmail.com 
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "PerformanceTimer.h"
#include <stdlib.h>

namespace glutFramework {
		
	PerformanceTimer::PerformanceTimer() {
		
	#ifdef WIN32	
		QueryPerformanceFrequency(&_freq);	// Retrieves the frequency of the high-resolution performance counter
		_start.QuadPart = 0;
		_end.QuadPart = 0;
	#else
		_start.tv_sec = 0;
		_start.tv_usec = 0;
		_end.tv_sec = 0;
		_end.tv_usec = 0;
		
	#endif
		
		_isStopped = true;
	}

	PerformanceTimer::~PerformanceTimer() {
	}

	void PerformanceTimer::start() {
	#ifdef WIN32
		 QueryPerformanceCounter(&_start);	// Retrieves the current value of the high-resolution performance counter
	#else
		gettimeofday(&_start, NULL);		// Get the starting time
	#endif
		_isStopped = false;
	}

	void PerformanceTimer::stop() {
	#ifdef WIN32
		QueryPerformanceCounter(&_end);
	#else
		gettimeofday(&_end, NULL);
	#endif
		
		_isStopped = true;	
	}
		
	bool PerformanceTimer::isStopped() const {
		return _isStopped;
	}
	
	double PerformanceTimer::getElapsedMicroseconds() {
		double microSecond = 0;

		if(!_isStopped) {
	#ifdef WIN32
			QueryPerformanceCounter(&_end);
	#else
			gettimeofday(&_end, NULL);
	#endif
		}
		
	#ifdef WIN32
		if(_start.QuadPart != 0 && _end.QuadPart != 0) {
			microSecond = (_end.QuadPart - _start.QuadPart) * (1000000.0 / _freq.QuadPart);
		}
	#else 
		microSecond = (_end.tv_sec * 1000000.0 + _end.tv_usec) - (_start.tv_sec * 1000000.0 + _start.tv_usec);
	#endif
		
		return microSecond;
	}

	double PerformanceTimer::getElapsedMilliseconds() {
		return getElapsedMicroseconds() / 1000.0;
	}

	double PerformanceTimer::getElapsedSeconds() {
		return getElapsedMicroseconds() / 1000000.0;
	}

}
