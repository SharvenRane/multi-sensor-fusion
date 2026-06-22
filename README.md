# multi-sensor-fusion

Fuse two sensor streams into one prediction, and show that fusion buys you
something a single sensor never could.

The setup imagines two aligned feature streams off a robot or a car. Call them
sensor A (say camera features) and sensor B (say depth or radar vectors). On
their own each stream is half blind. The thing you actually want to predict
depends on how the two streams agree or disagree, so you only get the answer by
looking at both at once. This repo builds that situation on purpose, trains
early fusion and late fusion models against single sensor baselines, and checks
that the fusion models win.

## Why a single sensor cannot win here

The label is the exclusive or of two hidden votes, one vote per stream. Each
stream casts its vote through a fixed linear readout of an informative
subspace, so the vote is a real thing a model has to learn, not noise. But XOR
has a property that makes it the perfect stress test for fusion: if you hide
one of the two inputs, the label is exactly 50/50 given the other. There is no
clever model, however large, that recovers an XOR output from one of its two
inputs alone. So any single sensor model is pinned at chance by construction,
while a model that sees both streams can solve the task cleanly.

To keep the per stream vote learnable in a few seconds on a CPU, the signal
lives in the first few coordinates of each stream and the rest is distractor
noise, and samples that sit right on a stream's decision boundary are dropped
so the learned boundary generalises instead of memorising edge cases. Those are
data hygiene choices. The hard part, the cross modal XOR, is untouched.

## The two fusion styles

Early fusion concatenates the raw streams and encodes the joint vector, so the
interaction between modalities is visible from the first layer.

Late fusion encodes each stream on its own, then concatenates the two
embeddings and runs a small fusion head. Because each branch stands alone, late
fusion can also run when a sensor drops out. The missing stream's embedding is
swapped for a learned stand in vector, so the head still receives a well formed
input and returns a prediction rather than crashing. On this XOR task the
prediction it returns with one stream gone is correctly back near chance, which
is the honest answer, the missing half of the signal really is gone.

## Results from one run

These came out of `python demo.py` on CPU. Training is stochastic so the exact
figures shift a little between runs, but the ordering holds every time.

```
train rows: 3574    test rows: 1711

sensor A only                0.489
sensor B only                0.486
early fusion                 0.924
late fusion (both)           0.959
late fusion (B missing)      0.482
late fusion (A missing)      0.509
```

Both single sensors sit at chance. Both fusion models clear 0.9 on held out
data. Take a stream away from the late fusion model and it falls back to chance,
exactly as it should when the information needed for the answer is no longer
present.

## Layout

```
src/
  data.py     synthetic aligned dataset, label = XOR of two per stream votes
  models.py   SingleSensorModel, EarlyFusionModel, LateFusionModel
  train.py    shared training loop, accuracy, per model forward adapters
tests/        pytest behaviour tests
demo.py       trains every variant once and prints the table above
```

## Running it

Install the dependencies, then run the tests and the demo.

```
pip install -r requirements.txt
pytest tests/ -q
python demo.py
```

## What the tests check

The tests are behaviour checks, not snapshots of a magic number.

- The dataset really does encode XOR: with no label noise and no margin
  filtering the label is exactly the XOR of the two recomputed votes, and no
  single raw feature of one stream tracks the label.
- The single sensor models stay near chance after training.
- Both the early fusion and the late fusion model beat each single sensor by a
  clear margin and clear 0.8 held out accuracy.
- The late fusion model runs with a modality missing, swapping in its learned
  stand in embedding, and its full pair prediction beats both degraded cases.

Everything runs on CPU with tiny tensors. There are no downloads and no API
keys.
