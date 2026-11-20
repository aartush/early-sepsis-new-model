"""Reproduce the four hybrid DL arms (their DL_balanced2.py).

Their training protocol, reproduced exactly:
  - 40 balanced datasets, trained *sequentially*, weights carried forward
  - one epoch per chunk in their setup, epochs configurable (paper implies more)
  - EarlyStopping(val_loss, patience=5) and ModelCheckpoint(save_best_only)
  - validation sets used only for monitoring; no re-randomisation per chunk
  - the final checkpoint is applied to all 40 test sets

Architectures are lifted from their build_model() so shapes and block order
match. `keras-core` is replaced by Keras 3 on the torch backend, which is the
maintained equivalent; this is noted in the reproduction report as a deviation.

Usage:
    py -m _reproduction.dl --model LSTM --try 1 --epochs 20
    py -m _reproduction.dl --all --epochs 20
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from _reproduction.common import (MODELS, N_SPLITS, PREDICTIONS, RESULTS,
                                  evaluate, read, save_json, summarise)

OBS_WIN = 24
N_STATIC = 10
LR = 0.001
BATCH = 32


def _keras():
    import os
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    import keras
    return keras


def load_chunk(idx: int):
    """Mirror of their load_data(idx) but fully path-resolved.

    Their code reads labels from `train_X_{i}.csv` (a file that no longer ships)
    while features come from `train_X_{i}_norm.csv`, which itself already carries
    `label` and `index` as the last two columns. We take labels from the norm
    file's label column, which is the same source the balanced split used, so
    the join is consistent and no label source is invented.
    """
    out = {}
    for role in ('train', 'val', 'test'):
        X = read('data_processed', f'{role}_X_{idx}_norm.csv')
        y = X['label'].to_numpy()
        feats = X.iloc[:, :-2]          # drop label + index
        out[role] = (feats.iloc[:, :-N_STATIC], feats.iloc[:, -N_STATIC:], y)
    return out


def to_arrays(temporal: pd.DataFrame, static: pd.DataFrame):
    n_feat = temporal.shape[1] // OBS_WIN
    return (temporal.to_numpy().reshape(len(temporal), OBS_WIN, n_feat).astype('float32'),
            static.to_numpy().astype('float32'))


def build_model(name: str):
    keras = _keras()
    from keras import Input, Model
    from keras.layers import (LSTM, BatchNormalization, Concatenate, Conv1D, Dense,
                              Dropout, Flatten, GlobalAveragePooling1D, MaxPool1D,
                              MultiHeadAttention)
    from keras.optimizers import Adam

    temporal_input = Input(shape=(OBS_WIN, 16), name='input_temporal')
    static_input = Input(shape=(N_STATIC,), name='input_static')

    def common(x):
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        return Dropout(0.2)(x)

    x_s = common(static_input)
    t = temporal_input

    if name == 'LSTM':
        t = LSTM(32, return_sequences=True)(t)
        t = BatchNormalization()(t)
        t = LSTM(32, return_sequences=True)(t)
        t = BatchNormalization()(t)
        t = MultiHeadAttention(num_heads=3, key_dim=4)(t, t)
        t = t + Dense(32)(temporal_input)
        t = GlobalAveragePooling1D()(t)

    elif name == '1DCNN':
        for _ in range(3):
            t = Conv1D(32, 3, activation='relu')(t)
            t = MaxPool1D(2)(t)
        t = MultiHeadAttention(num_heads=4, key_dim=4)(t, t)
        t = t + Conv1D(32, kernel_size=1, padding='same')(temporal_input)
        t = Flatten()(t)

    elif name == 'TCN':
        for _ in range(3):
            t = Conv1D(16, 3, activation='relu', dilation_rate=2, padding='causal')(t)
            t = BatchNormalization()(t)
        t = MultiHeadAttention(num_heads=3, key_dim=4)(t, t)
        t = Flatten()(t)

    elif name == '1DCNN-LSTM':
        t = Conv1D(32, 3, activation='relu')(t); t = MaxPool1D(2)(t)
        t = Conv1D(8, 3, activation='relu')(t);  t = MaxPool1D(2)(t)
        t = Conv1D(8, 3, activation='relu')(t);  t = MaxPool1D(2)(t)
        t = LSTM(32, return_sequences=True)(t)
        t = BatchNormalization()(t)
        t = MultiHeadAttention(num_heads=3, key_dim=4)(t, t)
        t = t + Conv1D(32, kernel_size=1, padding='same')(temporal_input)
        t = GlobalAveragePooling1D()(t)

    else:
        raise ValueError(name)

    x = Concatenate()([t, x_s])
    x = Dense(32, activation='relu')(x); x = BatchNormalization()(x)
    x = Dense(8, activation='relu')(x);  x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[temporal_input, static_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=LR),
                  loss='binary_crossentropy', metrics=['AUC'])
    return model


def run_arm(name: str, try_no: int, epochs: int, limit: int | None = None):
    keras = _keras()
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    t0 = time.time()
    n = limit or N_SPLITS
    model = build_model(name)
    model.summary(print_fn=lambda s: None)      # silence; count params below

    ckpt = MODELS / f'{name}_{try_no}.keras'
    history_rows = []

    for idx in range(1, n + 1):
        d = load_chunk(idx)
        Xt, Xs = to_arrays(d['train'][0], d['train'][1])
        Vt, Vs = to_arrays(d['val'][0], d['val'][1])
        y_tr, y_va = d['train'][2], d['val'][2]

        h = model.fit([Xt, Xs], y_tr,
                      validation_data=([Vt, Vs], y_va),
                      epochs=epochs, batch_size=BATCH, verbose=0,
                      callbacks=[
                          ModelCheckpoint(str(ckpt), monitor='val_loss',
                                          save_best_only=True, mode='min'),
                          EarlyStopping(monitor='val_loss', patience=5,
                                        restore_best_weights=True),
                      ])
        history_rows.append({
            'chunk': idx,
            'train_loss': float(h.history['loss'][-1]),
            'train_auc': float(h.history.get('AUC', h.history.get('auc', [np.nan]))[-1]),
            'val_loss': float(h.history['val_loss'][-1]),
            'val_auc': float(h.history.get('val_AUC', h.history.get('val_auc', [np.nan]))[-1]),
            'epochs_run': len(h.history['loss']),
        })
        print(f"  [{name} try{try_no}] chunk {idx:2d}/{n}  "
              f"val_loss={history_rows[-1]['val_loss']:.4f}  "
              f"val_auc={history_rows[-1]['val_auc']:.4f}  ({time.time()-t0:.0f}s)")

    # Final checkpoint applied to every balanced test set (their predict())
    eval_rows, probs = [], {}
    final = keras.models.load_model(ckpt, compile=False)
    final.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    for idx in range(1, n + 1):
        d = load_chunk(idx)
        Tt, Ts = to_arrays(d['test'][0], d['test'][1])
        p = final.predict([Tt, Ts], verbose=0).ravel()
        probs[idx] = p
        m = evaluate(p, d['test'][2])
        m['split'] = idx
        eval_rows.append(m)

    pd.DataFrame(probs).to_csv(PREDICTIONS / f'{name}_{try_no}_balanced_prob.csv', index=False)
    pd.DataFrame(eval_rows).to_csv(RESULTS / f'dl_{name}_{try_no}_per_split.csv', index=False)
    pd.DataFrame(history_rows).to_csv(RESULTS / f'dl_{name}_{try_no}_history.csv', index=False)

    summary = summarise(eval_rows, 'test_')
    summary.update({'model': name, 'try': try_no, 'epochs': epochs,
                    'n_splits': len(eval_rows), 'seconds': round(time.time() - t0, 1),
                    'params': int(final.count_params())})
    save_json(summary, f'dl_{name}_{try_no}_summary.json')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='LSTM',
                    choices=['LSTM', '1DCNN', 'TCN', '1DCNN-LSTM'])
    ap.add_argument('--try', dest='try_no', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()

    arms = ['LSTM', '1DCNN', 'TCN', '1DCNN-LSTM'] if a.all else [a.model]
    for arm in arms:
        print(f"\n=== {arm} (epochs={a.epochs}) ===")
        s = run_arm(arm, a.try_no, a.epochs, a.limit)
        print(f"  AUC={s['test_auc']:.4f}  SEN@90={s['test_sen_90']:.4f} "
              f"SPE@90={s['test_spec_90']:.4f}  params={s['params']}")
