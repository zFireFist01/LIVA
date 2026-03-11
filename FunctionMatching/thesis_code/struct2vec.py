from itertools import groupby
from pathlib import PosixPath, Path
from random import sample, shuffle, randint
from json import dump

import tensorflow as tf
import numpy as np
import sys
import gc


T = 5
SIGMA_DEPTH = 2
NUM_EPOCHS = 175
EMBEDDING_SIZE = 128
BATCH_SIZE = 32
TRAINING_RATIO = 0.8
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2

POS_LABEL = -1
NEG_LABEL = 0



class FunctionData:
    def __init__(self, name: str, bin_origin: str, adj_matrix, attributes):
        self.name = name
        self.bin_origin = bin_origin.split(".gcc")[0].split(".clang")[0]
        self.adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
        self.attributes = tf.convert_to_tensor(attributes, dtype=tf.float32)


class BinaryData:
    def __init__(self, name: str, functions):
        self.name = name
        self.functions = functions



class HistoryDump(tf.keras.callbacks.History):
    def __init__(self, filename="history.json"):
        super(HistoryDump, self).__init__()
        
        self.filename = filename
        self.history = {"epoches": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["epoches"].append(epoch)
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        # Save the history to a JSON file
        with open(self.filename, "w") as f:
            dump(self.history, f)


class TruePosSim(tf.keras.metrics.Metric):
    def __init__(self, name="true_pos_mean", **kwargs):
        super(TruePosSim, self).__init__(name=name, **kwargs)
        self.similarity = self.add_weight(name="similarity", initializer="zeros")
        self.n = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute the similarity metric
        def add():
            self.n += 1
            self.similarity.assign(tf.squeeze((self.similarity + y_pred) / self.n))

        condition = tf.reduce_all(tf.equal(y_true, tf.ones_like(y_true)))
        tf.cond(condition, add, lambda: None)

    def result(self):
        return self.similarity


class TrueNegSim(tf.keras.metrics.Metric):
    def __init__(self, name="true_neg_mean", **kwargs):
        super(TrueNegSim, self).__init__(name=name, **kwargs)
        self.similarity = self.add_weight(name="similarity", initializer="zeros")
        self.n = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute the similarity metric
        def add():
            self.n += 1
            print(f"N TRUE_NEG: {self.n}")
            self.similarity.assign(tf.squeeze((self.similarity + y_pred) / self.n))

        condition = tf.reduce_all(tf.equal(y_true, tf.zeros_like(y_true)))
        tf.cond(condition, add, lambda: None)

    def result(self):
        return self.similarity


class GraphNetwork(tf.keras.Model):
    def __init__(self, hidden_dim=EMBEDDING_SIZE, output_dim=EMBEDDING_SIZE, T=T):
        """
        hidden_dim: Dimensionality for node representations (μ_v).
        output_dim: Dimensionality for the graph-level output.
        T: Number of message-passing iterations.
        """
        super(GraphNetwork, self).__init__()
        self.T = T
        
        # Linear transformation for node features: W1 * x
        self.dense_x_to_mu = tf.keras.layers.Dense(hidden_dim)
        
        # Sigma function: σ(l) = P1 * ReLU(P2 * l)
        # First transformation: P2 * l, with ReLU activation.
        self.sigma_dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        # Second transformation: P1, with no activation.
        self.sigma_dense2 = tf.keras.layers.Dense(hidden_dim)
        
        # Readout layer: W2 * (sum_v μ_v^T)
        self.readout = tf.keras.layers.Dense(output_dim)

        # Loss and metrics
        self.sum_loss = tf.keras.metrics.Sum(name="sum_loss")
        self.mean_loss = tf.keras.metrics.Mean(name="mean_loss")
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.7)
        self.auc_metric = tf.keras.metrics.AUC(name="auc")
        self.true_pos = TruePosSim(name="true_pos")
        self.true_neg = TrueNegSim(name="true_neg")

    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.sum_loss, self.mean_loss, self.accuracy, self.auc_metric, self.true_pos, self.true_neg]

    def call(self, adj_matrix, node_features):
        """
        inputs: tuple (x, A)
          - x: Tensor of shape (num_nodes, feature_dim), the node features.
          - A: Tensor of shape (num_nodes, num_nodes), the adjacency matrix where
               A[v, u] = 1 if u ∈ N(v) (neighbors of v), else 0.
        """
        num_nodes = tf.shape(node_features)[0]
        hidden_dim = self.dense_x_to_mu.units
        
        # Initialize μ_v = 0 for all nodes (shape: (num_nodes, hidden_dim))
        mu = tf.zeros((num_nodes, hidden_dim))
        
        # Precompute W1 * x for all nodes (this remains constant over iterations)
        x_transformed = self.dense_x_to_mu(node_features)
        
        for _ in range(self.T):
            # For each node v, compute l_v = sum_{u in N(v)} μ_u^(t-1)
            # This is efficiently computed as a matrix multiplication A * mu.
            l = tf.matmul(adj_matrix, mu)
            
            # Compute σ(l) = P1 * ReLU(P2 * l)
            sigma_l = self.sigma_dense2(self.sigma_dense1(l))
            
            # Update node representations: μ_v^t = tanh(W1 * x_v + σ(l_v))
            mu = tf.tanh(x_transformed + sigma_l)
        
        # Graph readout: sum the node representations and apply the final transformation W2.
        graph_sum = tf.reduce_sum(mu, axis=0, keepdims=True)    # Shape: (1, hidden_dim)
        graph_embedding = self.readout(graph_sum)               # Shape: (1, output_dim)
        return graph_embedding
    

    def train_step(self, data):
        am1, a1, am2, a2, y_true = data

        with tf.GradientTape() as tape:
            f1_embed = self(am1, a1)
            f2_embed = self(am2, a2)
            sim = tf.losses.cosine_similarity(f1_embed, f2_embed, axis=1)
            loss = tf.square(sim - y_true)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if "loss" in metric.name:
                metric.update_state(loss)
            else:
                metric.update_state(y_true=-y_true, y_pred=(1-sim)/2)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    

    def test_step(self, data):
        am1, a1, am2, a2, y_true = data

        f1_embed = self(am1, a1)
        f2_embed = self(am2, a2)
        sim = tf.losses.cosine_similarity(f1_embed, f2_embed, axis=1)
        loss = tf.square(sim - y_true)

        # Update metrics
        for metric in self.metrics:
            if "loss" in metric.name:
                metric.update_state(loss)
            else:
                metric.update_state(-y_true, (1-sim)/2)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}





def parse_file(func_file: PosixPath) -> list[FunctionData]:

    with np.load(func_file, 'r', allow_pickle=True) as file:
        # extract data from file per each binary
        name = file['name']
        origin = file['origin']
        graph_repr = file['graph_repr']
        attributes = file['attributes']

        return FunctionData(str(name), str(origin), graph_repr, attributes)

def generate_positive_pairs(functions: list[FunctionData]) -> list[tuple[FunctionData, FunctionData, int]]:
    dataset = []
    for i in range(0, len(functions)-1):
        func_1 = functions[i]

        for j in range(i+1, len(functions)):
            func_2 = functions[j]
            dataset.append((func_1, func_2, POS_LABEL))
    
    return dataset
    

if __name__ == "__main__":
    dataset_dir = Path(sys.argv[1])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Can't find '{dataset_dir}'")
    
    # parsing binaries from the dataset files
    functions: list[FunctionData] = []
    for f in dataset_dir.iterdir():
        func = parse_file(f)
        if func is not None:
            functions.append(parse_file(f))

    functions = sorted(functions, key=lambda x: x.bin_origin + '.' + x.name)
    group_functions = [list(g) for _, g in groupby(functions, key=lambda x: x.bin_origin+'.'+x.name)]
    shuffle(group_functions)

    pair_dataset: list[tuple[FunctionData, FunctionData, int]] = []
    for group in group_functions:
        dataset = generate_positive_pairs(group)
        pair_dataset.extend(dataset)

    print(f"Generated {len(pair_dataset)} positive pairs")

    negative_pairs = len(pair_dataset)
    for group in group_functions:
        shuffle(group)

    seen = set()
    neg_pairs = []

    group_indices = list(range(len(group_functions)))
    while negative_pairs > 0:
        j, k = sample(group_indices, k=2)
        g1, g2 = group_functions[j], group_functions[k]
        if j == k or g1[0].name.split('.')[0] == g2[0].name.split('.')[0]:
            continue

        # extract two functions from the group
        func_1 = g1[randint(0, len(g1)-1)]
        func_2 = g2[randint(0, len(g2)-1)]

        pair = (func_1, func_2, NEG_LABEL) if j < k else (func_2, func_1, NEG_LABEL)
        if pair in seen:
            continue
        seen.add(pair)
        neg_pairs.append(pair)
        negative_pairs -= 1

    pair_dataset.extend(neg_pairs)
    del neg_pairs, seen, group_indices

    print(f"Generated {len(pair_dataset)} pairs")
    print(F"Generated {len([p for p in pair_dataset if p[2] == POS_LABEL])} positive pairs")
    print(f"Generated {len([p for p in pair_dataset if p[2] == NEG_LABEL])} negative pairs")

    # divide the dataset 
    train_size = int(len(pair_dataset) * TRAINING_RATIO)
    val_size = len(pair_dataset) - train_size
    
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")

    shuffle(pair_dataset)
    train_dataset = pair_dataset[:train_size]
    validation_dataset = pair_dataset[train_size:]
    
    del pair_dataset, functions, group_functions
    gc.collect()

    print(f"Train dataset size: {train_size}, pos: {len([p for p in train_dataset if p[2] == POS_LABEL])}, neg: {len([p for p in train_dataset if p[2] == NEG_LABEL])}")
    print(f"Validation dataset size: {val_size}, pos: {len([p for p in validation_dataset if p[2] == POS_LABEL])}, neg: {len([p for p in validation_dataset if p[2] == NEG_LABEL])}")

    train_dataset = (
        (f1.adj_matrix, f1.attributes,
         f2.adj_matrix, f2.attributes, y_true)
        for f1, f2, y_true in train_dataset * NUM_EPOCHS
    )

    validation_dataset = (
        (f1.adj_matrix, f1.attributes,
         f2.adj_matrix, f2.attributes, y_true)
        for f1, f2, y_true in validation_dataset * NUM_EPOCHS
    )


    # Convert to TensorFlow dataset (keeping separate tensors)
    tf_train_dataset = tf.data.Dataset.from_generator(
        lambda: iter(train_dataset),
        output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),    # am1 (adjacency matrix)
            tf.TensorSpec(shape=(None, 128), dtype=tf.float32),     # attr1 (node features)
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),    # am2
            tf.TensorSpec(shape=(None, 128), dtype=tf.float32),     # attr2
            tf.TensorSpec(shape=(), dtype=tf.float32)               # y_true (label)
        )
    ).shuffle(buffer_size=train_size)

    tf_validation_dataset = tf.data.Dataset.from_generator(
        lambda: iter(validation_dataset),
        output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),    # am1 (adjacency matrix)
            tf.TensorSpec(shape=(None, 128), dtype=tf.float32),     # attr1 (node features)
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),    # am2
            tf.TensorSpec(shape=(None, 128), dtype=tf.float32),     # attr2
            tf.TensorSpec(shape=(), dtype=tf.float32)               # y_true (label)
        )
    )

    # Instantiate the model.
    model = GraphNetwork(512, 128, 5)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    # Define checkpoint path
    checkpoint_path = sys.argv[2]

    # Create a callback to save the best model based on validation loss
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_mean_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )
    history_dump = HistoryDump(filename=sys.argv[3])

    results = model.fit(tf_train_dataset ,batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=tf_validation_dataset, steps_per_epoch=train_size//BATCH_SIZE, validation_steps=val_size//BATCH_SIZE,callbacks=[checkpoint_callback, history_dump])
