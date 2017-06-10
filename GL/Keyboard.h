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

#ifndef KEYBOARD_H
#define KEYBOARD_H

/** Keyboard.h 
 *  
 * Description: A basic utility class to keep track of what keys are currently
 *	kbeing pressed.
 *
 * Author: Paul Solt 8-21-10
 */

namespace glutFramework {
	
	class Keyboard {
	private:
		enum Key { UP, DOWN, RELEASED };
		static const int NUMBER_KEYS = 256;
		Key keys[ NUMBER_KEYS ];

	public:

		/** Name: Keyboard()
		 * 
		 * Description: Initialize all keys in the up state 
		 */
		Keyboard();

		/** Name: keyDown()
		 *
		 * Description: Set the key to the down state 
		 * Param: key - the key that is being pressed
		 */
		void keyDown( int key );

		/** Name: keyDown()
		 *
		 * Description: Set the key to the up state 
		 * Param: key - the key that is being released
		 */
		void keyUp( int key );

		/** Name: isKeyDown
		 *
		 * Description: Test to see if the key is being pressed
		 */
		bool isKeyDown( int key );
	};

}
#endif
