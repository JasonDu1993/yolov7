import numpy as np

def get_split_set(total_num, thread_num):
  avg_num = total_num // thread_num
  split_set_list = []

  for i in range(thread_num):
    start = i * avg_num
    end = (i + 1) * avg_num

    if i == thread_num - 1:
      end = total_num
    split_set_list.append((start, end))

  return split_set_list

