from libc.stdlib cimport malloc


cdef IterParams* define_iter_params(size_t[2] shape,
                                    size_t[2] window_size,
                                    double fraction_accepted,
                                    bint reduce) noexcept nogil:
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
