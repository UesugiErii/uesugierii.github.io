# 尝试随机生成数据, 测试聚类后中心小于目标值的概率
# 未加速初始版本
import pickle
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf

num_interest = 4
seq_len = 20
batch_size = 128

# 有问题的数据
with open('vector.pkl', 'rb') as f:
    d = pickle.load(f)
d = d[None, :, :]
mask = np.ones((1, 20), dtype=np.float32)
mask[0, 9:] = 0

total = 0
found = 0

while 1:
    # d = np.random.random((batch_size, seq_len, 64)).astype(np.float32)
    # mask = np.ones((batch_size, seq_len), dtype=np.float32)
    # mask[:, np.random.randint(5, seq_len):] = 0

    item_his_eb = d

    num_segments = 1 + num_interest  # 几个聚类中心, 开始的1对应值0是给padding的
    centers = item_his_eb[:, :num_interest, :]  # bs, self.num_interest, dim

    # belong 代表每行数据中每个物品属于第几个聚类
    last_belong = tf.zeros((item_his_eb.shape[0], seq_len), dtype=tf.int32)
    belong = tf.ones((item_his_eb.shape[0], seq_len), dtype=tf.int32)

    iter_n = 0

    while tf.reduce_any(tf.not_equal(last_belong, belong)):
        distance = tf.reduce_sum(
            tf.math.square(
                tf.tile(item_his_eb[:, :, None, :], [1, 1, num_interest, 1])
                -
                tf.tile(centers[:, None, :, :], [1, seq_len, 1, 1])
            ),
            axis=-1
        )

        last_belong = belong

        belong = tf.argmin(distance, axis=-1, output_type=tf.dtypes.int32) + 1
        belong = tf.multiply(belong, tf.cast(mask, belong.dtype))

        num_rows = tf.shape(belong)[0]
        rows_idx = tf.range(num_rows)
        segment_ids_per_row = belong + num_segments * tf.expand_dims(rows_idx, axis=1)

        centers = tf.math.unsorted_segment_mean(item_his_eb, segment_ids_per_row, num_segments * num_rows)
        centers = tf.reshape(centers, (num_rows, num_segments, item_his_eb.shape[-1]))

        centers = centers[:, 1:, :]

        iter_n += 1

        # print(belong)

    belong = tf.stop_gradient(belong)

    for ni in range(1, num_interest+1):
        local_mask = tf.equal(belong, ni)
        found += local_mask.shape[0] - tf.math.reduce_sum(tf.cast(tf.math.reduce_any(local_mask, axis=1),tf.int32), axis=0)

    total += 1

    print(found, total)  # 发生这种情况的概率
