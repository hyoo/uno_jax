import argparse
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Iterable, Callable
from data import load
from timeit import default_timer as timer
from clu import parameter_overview


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/uno_top21_auc', help='Datafile prefix')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='Epochs')
    parser.add_argument('--batch_size', '-z', default=32, type=int, help='Batch Size')
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float, help='Learning Rate')

    args, unparsed = parser.parse_known_args()
    return args, unparsed


class GeneFeatureModel(nn.Module):
    input_size: int
    dense_layers: Iterable
    dropout_rate: float
    kernel_init: Callable = jax.nn.initializers.glorot_normal()

    @nn.compact
    def __call__(self, z):
        for i, n in enumerate(self.dense_layers):
            z = nn.Dense(n, kernel_init=self.kernel_init, name=f'dense_{i}')(z)
            z = nn.relu(z)
            if self.dropout_rate > 0.:
                z = nn.Dropout(rate=self.dropout_rate, deterministic=False)(z)
        return z


class DrugFeatureModel(nn.Module):
    input_size: int
    dense_layers: Iterable
    dropout_rate: float
    kernel_init: Callable = jax.nn.initializers.glorot_normal()

    @nn.compact
    def __call__(self, z):
        for i, n in enumerate(self.dense_layers):
            z = nn.Dense(n, kernel_init=self.kernel_init, name=f'dense_{i}')(z)
            z = nn.relu(z)
            if self.dropout_rate > 0.:
                z = nn.Dropout(self.dropout_rate, deterministic=False)(z)
        return z


class UnoModel(nn.Module):
    """ Uno Model """
    gene_input_size: int = 942
    gene_dense_layers: Iterable = (1000, 1000, 1000)
    drug_input_size: int = 5270
    drug_dense_layers: Iterable = (1000, 1000, 1000)
    dense_layers: Iterable = (1000, 1000, 1000, 1000, 1000)
    dropout_rate: float = 0.1
    kernel_init: Callable = jax.nn.initializers.glorot_normal()

    def setup(self):
        self.gene_net = GeneFeatureModel(
                input_size=self.gene_input_size,
                dense_layers=self.gene_dense_layers,
                dropout_rate=self.dropout_rate,
                kernel_init=self.kernel_init,
        )
        self.drug_net = DrugFeatureModel(
                input_size=self.drug_input_size,
                dense_layers=self.drug_dense_layers,
                dropout_rate=self.dropout_rate,
                kernel_init=self.kernel_init,
        )

    @nn.compact
    def __call__(self, inputs):
        gene, drug = inputs
        gene = self.gene_net(gene)
        drug = self.drug_net(drug)
        x = jax.lax.concatenate([gene, drug], dimension=1)

        for i, n in enumerate(self.dense_layers):
            x = nn.Dense(n, kernel_init=self.kernel_init, name=f'dense_{i}')(x)
            x = nn.relu(x)
            if self.dropout_rate > 0.:
                x = nn.Dropout(self.dropout_rate, deterministic=False)(x)

        x = nn.Dense(1)(x)
        return x


@jax.jit
def apply_model(state, inputs, target, do_rng):
    def loss_fn(params):
        logits = UnoModel().apply({'params': params}, inputs, rngs={'dropout': do_rng})
        min_norm = 1e-8
        loss = jnp.maximum(jnp.sqrt(jnp.mean((logits - target)**2)), min_norm)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    return grads, loss


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, steps_per_epoch, rng, do_rng):
    epoch_loss = []

    for step, batch in enumerate(train_ds):
        if step >= steps_per_epoch:
            break
        (batch_gene, batch_drug), batch_target = batch
        batch_gene = jnp.array(batch_gene)
        batch_drug = jnp.array(batch_drug)
        batch_target = jnp.array(batch_target)

        grads, loss = apply_model(state, (batch_gene, batch_drug), batch_target, do_rng)
        state = update_model(state, grads)
        epoch_loss.append(loss)
    train_loss = np.mean(epoch_loss)
    return state, train_loss


def evaluate_epoch(state, val_ds, do_rng):
    epoch_loss = []
    for _, batch in enumerate(val_ds):
        (batch_gene, batch_drug), batch_target = batch
        batch_gene = jnp.array(batch_gene)
        batch_drug = jnp.array(batch_drug)
        batch_target = jnp.array(batch_target)

        _, loss = apply_model(state, (batch_gene, batch_drug), batch_target, do_rng)
        epoch_loss.append(loss)
    val_loss = np.mean(epoch_loss)
    return val_loss


def main():
    args, _ = parse_arguments()

    # total records: 423952, 52994, 52994
    train_ds, val_ds = load(file_prefix=args.data, batch_size=args.batch_size)
    steps_per_epoch = (423952 + args.batch_size) // args.batch_size

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    dropout_rng = jax.random.PRNGKey(1)
    dropout_rng, init_dropout_rng = jax.random.split(dropout_rng)
    init_rngs = {'params': init_rng, 'dropout': init_dropout_rng}

    model = UnoModel()
    _model = model.init(init_rngs, (jnp.ones([1, 942]), jnp.ones([1, 5270])))
    print(parameter_overview.get_parameter_overview(_model))

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=_model['params'],
        tx=optax.adamw(learning_rate=args.learning_rate),
    )

    for epoch in range(1, (args.epochs + 1)):
        rng, input_rng = jax.random.split(rng)
        dropout_rng, input_dropout_rng = jax.random.split(dropout_rng)
        start_t = timer()
        state, train_loss = train_epoch(state, train_ds, steps_per_epoch, input_rng, input_dropout_rng)
        val_loss = evaluate_epoch(state, val_ds, input_dropout_rng)
        elapsed = timer() - start_t

        print(
            f'epoch: {epoch:3d}, elapsed:{elapsed:.1f}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}'
        )


if __name__ == '__main__':
    main()
