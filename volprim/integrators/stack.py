# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides a dynamic stack implementation using the Dr.Jit library.
"""

import drjit as _dr

def alloc_stack(Type, Index, alloc_size):
    """
    The `alloc_stack` function creates a stack class with methods to manage a stack
    data structure. The stack is parameterized by a data type (`Type`) and an index
    type (`Index`), and it supports operations such as push, pop, and resize.
    """
    class Stack:
        def __init__(self):
            self._local = _dr.alloc_local(Type, alloc_size)
            self._size = _dr.zeros(Index)

        DRJIT_STRUCT = { '_size' : Index, '_local': _dr.Local[Type] }

        def size(self):
            return Index(self._size)

        def alloc_size(self):
            return alloc_size

        def is_empty(self):
            return self._size == 0

        def resize(self, new_size, active=True):
            self._size = _dr.select(active, new_size, self._size)

        def clear(self, active=True):
            self._size[_dr.mask_t(Index)(active)] = 0

        def top(self):
            valid = self._size > 0
            index = _dr.select(valid, self._size - 1, 0)
            return _dr.select(valid, self._local[index], _dr.zeros(Type))

        def push(self, value, active=True):
            valid = active & (self._size < alloc_size)
            index = _dr.select(valid, self._size, 0)
            self._local[index] = _dr.select(active, value, self._local[index])
            self._size = _dr.select(active, self._size + 1, self._size)

        def pop(self, active=True):
            v = self.top()
            self._size = _dr.select(active & (self._size > 0), self._size - 1, self._size)
            return v

        def value(self, index, active=True):
            valid = active & (index < self._size)
            index = _dr.select(valid, index, 0)
            return _dr.select(valid, self._local[index], _dr.zeros(Type))

        def write(self, value, index, active=True):
            valid = active & (index < self._size)
            index = _dr.select(valid, index, 0)
            self._local[index] = _dr.select(valid, value, self._local[index])

        def __repr__(self):
            return f'Stack[type={Type}, size={self._size}, values={self._local}]'

    return Stack()
