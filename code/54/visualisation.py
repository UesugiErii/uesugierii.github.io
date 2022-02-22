# 可视化
import os
import tensorflow as tf
from tensorboard.plugins import projector
import pickle

with open('vector.pkl', 'rb') as f:
    d = pickle.load(f)
d = d[:9, :]

log_dir = '/home/zx/workspace/tfb'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save Labels separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    for i in range(d.shape[0]):
        f.write(f"{i}\n")

weights = tf.Variable(initial_value=d)

# Create a checkpoint from embedding, the filename and key are the
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)

# tensorboard --host 0.0.0.0 --logdir /home/zx/workspace/tfb
# http://{your_ip}:6006/#projector
