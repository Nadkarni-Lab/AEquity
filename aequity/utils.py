def generate_sample_sizes(max_sample_size : int = 5000, log_scale: int = 2, min_sample_size: int = 64, absolute_scale = None):
    sample_size_list = list()

    # print('Absolute scale', absolute_scale)
    if(absolute_scale == False):
        current_sample_size = min_sample_size
        while current_sample_size < max_sample_size:
            sample_size_list.append(current_sample_size)
            current_sample_size = current_sample_size * log_scale
    # if(absolute_scale == False):
    #     sample_size = int(max_sample_size)
    #     while sample_size > min_sample_size:
    #         sample_size_list.append(sample_size)
    #         sample_size = int(sample_size / log_scale)
    #     sample_size_list.append(min_sample_size)
    
    else:
        for sample_size in range(min_sample_size, max_sample_size, absolute_scale):
            sample_size_list.append(sample_size)
        sample_size_list.append(max_sample_size)
    sample_size_list.sort()
    # print(sample_size_list)
    return sample_size_list