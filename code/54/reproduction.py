# 尝试随机生成数据, 测试聚类后中心小于目标值的概率
# 3.py加速版本
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf

k = 4  # 最终结果的聚类中心数
n = 20  # 数据长度
batch_size = 20480


@tf.function
def f():
    data = tf.random.normal(shape=(batch_size, n, 64), mean=0.0, stddev=0.01, dtype=tf.dtypes.float32)

    len_ = tf.cast(tf.experimental.numpy.random.randint(7, n, dtype=tf.experimental.numpy.int32), tf.dtypes.int32)
    mask1 = tf.ones((batch_size, len_), dtype=np.float32)
    mask0 = tf.zeros((batch_size, n - len_), dtype=np.float32)
    mask = tf.concat([mask1, mask0], axis=-1)

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

        # print(belong)

    belong = tf.stop_gradient(belong)

    found = 0

    for ni in range(1, k + 1):
        local_mask = tf.equal(belong, ni)
        found += batch_size - tf.math.reduce_sum(
            tf.cast(tf.math.reduce_any(local_mask, axis=1), tf.int32),
            axis=0
        )

    return found


total = 0
total_found = 0

while 1:
    total += 1
    total_found += f()
    if total % 100 == 0:
        print(total_found, total)
