# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport isnan


cdef IterParams* define_iter_params(size_t[2] shape,
                                    size_t[2] window_size,
                                    double fraction_accepted,
                                    bint reduce):
    cdef IterParams* ip = <IterParams*> malloc(sizeof(IterParams))

    if reduce:
        ip.shape[0] = shape[0] // window_size[0]
        ip.shape[1] = shape[1] // window_size[1]
        ip.fringe[0] = 0
        ip.fringe[1] = 0
        ip.stop[0] = shape[0]
        ip.stop[1] = shape[1]
        ip.step[0] = window_size[0]
        ip.step[1] = window_size[1]

    else:
        ip.shape[0] = shape[0]
        ip.shape[1] = shape[1]
        ip.fringe[0] = window_size[0] // 2
        ip.fringe[1] = window_size[1] // 2
        ip.stop[0] = shape[0] - window_size[0] + 1
        ip.stop[1] = shape[1] - window_size[1] + 1
        ip.step[0] = 1
        ip.step[1] = 1

    ip.iter[0] = ip.stop[0] // ip.step[0]
    ip.iter[1] = ip.stop[1] // ip.step[1]
    ip.num_values = window_size[0] * window_size[1]
    ip.threshold = fraction_accepted * ip.num_values + 1

    return ip


cdef class Iterator:
    def __cinit__(self, double[:, ::1] a, np.uint8_t[:, ::1] mask, double fraction_accepted, bint reduce):
        cdef:
            size_t i, j

        self.reduce = reduce
        self.mask = mask
        self.shape[0] = a.shape[0]
        self.shape[1] = a.shape[1]

        self.num_values = 0

        self.window_shape[0] = mask.shape[0]
        self.window_shape[1] = mask.shape[1]

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                self.num_values += self.mask[i, j]

        self.threshold = fraction_accepted * self.num_values

        if reduce:
            self.output_shape[0] = a.shape[0] // mask.shape[0]
            self.output_shape[1] = a.shape[1] // mask.shape[1]
            self.fringe[0] = 0
            self.fringe[1] = 0
            self.stop[0] = a.shape[0]
            self.stop[1] = a.shape[1]
            self.step[0] = mask.shape[0]
            self.step[1] = mask.shape[1]

        else:
            self.output_shape[0] = a.shape[0]
            self.output_shape[1] = a.shape[1]
            self.fringe[0] = mask.shape[0] // 2
            self.fringe[1] = mask.shape[1] // 2
            self.stop[0] = a.shape[0] - mask.shape[0] + 1
            self.stop[1] = a.shape[1] - mask.shape[1] + 1
            self.step[0] = 1
            self.step[1] = 1

        self.xyiter[0] = self.stop[0] // self.step[0]
        self.xyiter[1] = self.stop[1] // self.step[1]
        self.iterations = self.xyiter[0] * self.xyiter[1]

        self.window = FocalWindow()
        self.window.values = <double*> malloc(self.num_values * sizeof(double))
        if self.window.values == NULL:
            raise MemoryError("window.values memory error")

    def __dealloc__(self):
        if self.window.values != NULL:
            free(self.window.values)

    cdef FocalWindow* get_window_values_v1(self, size_t iteration, double[:, ::1] a) noexcept nogil:
        cdef:
            size_t i, j, p, q, x, y

        self.window.skip = 0
        self.window.num_values = 0

        i = iteration // self.xyiter[1]
        j = iteration % self.xyiter[1]

        y = i * self.step[0]
        x = j * self.step[1]

        if not self.reduce and isnan(a[y + self.fringe[0], x + self.fringe[1]]):
            self.window.skip = 1
            return &self.window

        for p in range(self.window_shape[0]):
            for q in range(self.window_shape[1]):
                if not self.mask[p, q]:
                    continue
                if isnan(a[y + p, x + q]):
                    continue
                self.window.values[self.window.num_values] = a[y + p, x + q]
                self.window.num_values += 1

        if self.window.num_values == 0 or self.window.num_values < self.threshold:
            self.window.skip = 1

        return &self.window

    cdef void get_output_index(self, size_t iteration, size_t* yx) noexcept nogil:
        # todo; multiply by step
        cdef:
            size_t i, j
        i = iteration // self.xyiter[1]
        j = iteration % self.xyiter[1]
        yx[0] = i + self.fringe[0]
        yx[1] = j + self.fringe[1]

    def get_values(self, iteration: int, a: np.ndarray) -> np.ndarray:
        if not a.shape[0] == self.shape[0] or not a.shape[1] == self.shape[1]:
            raise ValueError(f"Shapes don't match: a.shape=({a.shape[0]}, {a.shape[1]}), {self.shape=}")

        if iteration < 0 or iteration >= self.iterations:
            raise ValueError(f"Expected iteration between 0 and {self.iterations}, got {iteration}")

        self.get_window_values_v1(iteration, a)

        if self.window.skip == 1:
            return np.array([], dtype=np.float64)

        return np.asarray([self.window.values[i] for i in range(self.window.num_values)])

    def get_properties(self) -> dict:
        return {
            "window_shape": (self.window_shape[0], self.window_shape[1]),
            "stop": (self.stop[0], self.stop[1]),
            "step": (self.step[0], self.step[1]),
            "shape": (self.shape[0], self.shape[1]),
            "output_shape": (self.output_shape[0], self.output_shape[1]),
            "fringe": (self.fringe[0], self.fringe[1]),
            "xyiter": (self.xyiter[0], self.xyiter[1]),
            "iterations": self.iterations,
            "num_values": self.num_values,
            "threshold": self.threshold,
            "reduce": self.reduce,
        }
