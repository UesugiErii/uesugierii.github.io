# 测试问题数据
import pickle
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf

with open('vector.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.shape)  # (20, 64)

data = data[None, :, :]
k = 4  # 最终结果的聚类中心数
n = 20  # 数据长度
mask = np.ones((1, 20), dtype=np.float32)
mask[0, 9:] = 0

data = tf.multiply(data, mask[:, :, None])

num_segments = 1 + k  # 几个聚类中心, 开始的1对应值0是给padding的(用mask0遮住的)
centers = data[:, :k, :]  # bs, k, dim

# belong 代表每行数据中每个物品属于第几个聚类
last_belong = tf.zeros((data.shape[0], n), dtype=tf.int32)
belong = tf.ones((data.shape[0], n), dtype=tf.int32)

iter_n = 0

while tf.reduce_any(tf.not_equal(last_belong, belong)):
    distance = tf.reduce_sum(
        tf.math.square(
            tf.tile(data[:, :, None, :], [1, 1, k, 1])
            -
            tf.tile(centers[:, None, :, :], [1, n, 1, 1])
        ),
        axis=-1
    )

    last_belong = belong

    belong = tf.argmin(distance, axis=-1, output_type=tf.dtypes.int32) + 1
    belong = tf.multiply(belong, tf.cast(mask, belong.dtype))

    num_rows = tf.shape(belong)[0]
    rows_idx = tf.range(num_rows)
    segment_ids_per_row = belong + num_segments * tf.expand_dims(rows_idx, axis=1)

    centers = tf.math.unsorted_segment_mean(data, segment_ids_per_row, num_segments * num_rows)
    centers = tf.reshape(centers, (num_rows, num_segments, data.shape[-1]))

    centers = centers[:, 1:, :]

    iter_n += 1

    print(belong)

# tf.Tensor([[1 2 3 4 2 2 1 2 4 0 0 0 0 0 0 0 0 0 0 0]], shape=(1, 20), dtype=int32)
# tf.Tensor([[1 3 3 1 3 4 1 4 4 0 0 0 0 0 0 0 0 0 0 0]], shape=(1, 20), dtype=int32)
# tf.Tensor([[1 3 3 1 3 4 1 4 4 0 0 0 0 0 0 0 0 0 0 0]], shape=(1, 20), dtype=int32)
