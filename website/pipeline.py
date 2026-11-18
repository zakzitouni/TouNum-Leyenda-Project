"""The Leyenda workflow for one image, with both saved captioners.

Binary classification (photo / not photo), then pre-processing (denoising) of the photos, then captioning.
The model code is copied from the notebook so that the captions are the same as in the report.
"""
import io, json, math
from pathlib import Path
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from PIL import Image, ImageOps

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=5500)])

PROJECT = Path(__file__).resolve().parents[1]                 # the repository root
CLASSIFIER_DIR, DENOISER_DIR, CAPTIONER_DIR = (PROJECT / "1 - Binary Classifier", PROJECT / "2 - Denoiser",
                                               PROJECT / "3 - Captioner")
CNN_WEIGHTS = CAPTIONER_DIR / "cnn_weights"                     # optional local copy of the ImageNet weights of both CNNs
D1_MODEL = CLASSIFIER_DIR / "modele_photo.keras"
D2_MODEL = DENOISER_DIR / "denoiser_coco_v2.keras"              # DenoiserNotebook.ipynb, Section 13
CAPTIONERS = {"gru": CAPTIONER_DIR / "leyenda_captioner_gru",
              "transformer": CAPTIONER_DIR / "leyenda_captioner"}

# Default order, chosen per request on the page. False (1-2-3, the brief): the classifier sees the image as received and
# only photographs are denoised. True (2-1-3): every image is denoised before the classifier (CaptionerNotebook.ipynb, Extension E2:
# more noisy photographs kept).
DENOISE_FIRST = False
MAX_SIDE, PHOTO_THRESHOLD, TILE, MARGIN = 1024, 0.5, 256, 64
PAD, UNK, START, END = 0, 1, 2, 3


# ---------------- Deliverable 3, InceptionV3 + Up-Down GRU (notebook Section 6)
class Encoder(keras.Model):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = layers.LayerNormalization(); self.fc = layers.Dense(dim); self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        return self.drop(tf.nn.relu(self.fc(self.norm(x))), training=training)


class Attention(keras.Model):
    def __init__(self, units):
        super().__init__(); self.W1 = layers.Dense(units); self.W2 = layers.Dense(units); self.V = layers.Dense(1)

    def call(self, features, keys, hidden):
        a = tf.nn.softmax(self.V(tf.nn.tanh(keys + self.W2(hidden)[:, None])), axis=1)
        return tf.reduce_sum(a * features, axis=1), a


class UpDownDecoder(keras.Model):
    def __init__(self, emb, units, vocab, dropout=0.5):
        super().__init__(); self.units = units
        self.embedding = layers.Embedding(vocab, emb)
        self.c1 = layers.GRUCell(units); self.c2 = layers.GRUCell(units)
        self.attention = Attention(units); self.drop = layers.Dropout(dropout)
        self.fc = layers.Dense(emb)

    def initial_state(self, n):
        z = tf.zeros((n, self.units)); return (z, z)

    def prepare(self, features):
        return self.attention.W1(features), tf.reduce_mean(features, axis=1)

    def step(self, word, features, prepared, state, training=False):
        keys, pooled = prepared; h1, h2 = state
        h1, _ = self.c1(tf.concat([h2, pooled, self.embedding(word)], -1), h1)
        ctx, a = self.attention(features, keys, h1)
        h2, _ = self.c2(tf.concat([ctx, self.drop(h1, training=training)], -1), h2)
        return h2, (h1, h2), a

    def logits(self, out, training=False):
        return tf.matmul(self.fc(self.drop(out, training=training)), self.embedding.embeddings, transpose_b=True)


def build_gru(cfg, V):
    return Encoder(cfg["enc"], cfg["enc_drop"]), UpDownDecoder(cfg["emb"], cfg["units"], V, cfg["dropout"])


# ---------------- Deliverable 3, ConvNeXt-XL + Transformer (notebook Extension E4)
class Attn(layers.Layer):
    def __init__(self, d, heads, dropout):
        super().__init__(); self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (layers.Dense(d) for _ in range(4)); self.drop = layers.Dropout(dropout)

    def split(self, x):
        return tf.transpose(tf.reshape(x, (tf.shape(x)[0], -1, self.h, self.dh)), (0, 2, 1, 3))

    def kv(self, m):
        return self.split(self.k(m)), self.split(self.v(m))

    def attend(self, x, k, v, mask=None, training=False):
        s = tf.matmul(self.split(self.q(x)), k, transpose_b=True) / math.sqrt(self.dh)
        if mask is not None:
            s = tf.where(mask, s, -1e9)
        a = tf.nn.softmax(s, -1); o = tf.matmul(self.drop(a, training=training), v)
        o = tf.reshape(tf.transpose(o, (0, 2, 1, 3)), (tf.shape(x)[0], -1, self.h * self.dh))
        return self.o(o), a

    def call(self, x, m, vis=None, training=False):
        k, v = self.kv(m); return self.attend(x, k, v, vis, training)


def ffn(d, ff, dropout):
    return keras.Sequential([layers.Dense(ff, activation="gelu"), layers.Dropout(dropout), layers.Dense(d)])


class EncLayer(layers.Layer):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.n1, self.n2 = layers.LayerNormalization(), layers.LayerNormalization()
        self.att = Attn(d, heads, dropout); self.ff = ffn(d, ff, dropout); self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        h = self.n1(x); x = x + self.drop(self.att(h, h, training=training)[0], training=training)
        return x + self.drop(self.ff(self.n2(x), training=training), training=training)


class DecLayer(layers.Layer):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.n1, self.n2, self.n3 = (layers.LayerNormalization() for _ in range(3))
        self.self_att, self.cross = Attn(d, heads, dropout), Attn(d, heads, dropout)
        self.ff = ffn(d, ff, dropout); self.drop = layers.Dropout(dropout)

    def call(self, x, cross_kv, self_kv=None, vis=None, training=False):
        h = self.n1(x); k, v = self.self_att.kv(h) if self_kv is None else self_kv
        x = x + self.drop(self.self_att.attend(h, k, v, vis, training)[0], training=training)
        c, a = self.cross.attend(self.n2(x), *cross_kv, training=training)
        x = x + self.drop(c, training=training)
        return x + self.drop(self.ff(self.n3(x), training=training), training=training), a


class TEncoder(keras.Model):
    def __init__(self, d, n_layers, heads, ff, dropout, regions):
        super().__init__()
        self.norm = layers.LayerNormalization(); self.fc = layers.Dense(d); self.drop = layers.Dropout(dropout)
        self.pos = self.add_weight(shape=(regions, d), initializer=keras.initializers.RandomNormal(stddev=0.02), name="pos")
        self.blocks = [EncLayer(d, heads, ff, dropout) for _ in range(n_layers)]; self.out_norm = layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.drop(tf.nn.relu(self.fc(self.norm(x))), training=training) + self.pos
        for b in self.blocks:
            x = b(x, training=training)
        return self.out_norm(x)


class TDecoder(keras.Model):
    def __init__(self, d, n_layers, heads, ff, vocab, dropout, length):
        super().__init__(); self.d, self.length, self.heads = d, length, heads
        self.embedding = layers.Embedding(vocab, d)
        self.pos = self.add_weight(shape=(length, d), initializer=keras.initializers.RandomNormal(stddev=0.02), name="pos")
        self.blocks = [DecLayer(d, heads, ff, dropout) for _ in range(n_layers)]
        self.out_norm = layers.LayerNormalization(); self.drop = layers.Dropout(dropout)

    def embed(self, words_, pos, training=False):
        return self.drop(self.embedding(words_) * math.sqrt(self.d) + pos, training=training)

    def prepare(self, features):
        return tuple(t for b in self.blocks for t in b.cross.kv(features)), features

    def initial_state(self, n):
        z = tf.zeros((n, self.heads, self.length, self.d // self.heads))
        return (tf.zeros((n,), tf.int32),) + (z,) * (2 * len(self.blocks))

    def step(self, word, features, prepared, state, training=False):
        pos, cache = state[0], state[1:]; cross = prepared[0]
        x = self.embed(word[:, None], tf.gather(self.pos, pos)[:, None], training)
        put = tf.one_hot(pos, self.length)[:, None, :, None]
        visible = tf.sequence_mask(pos + 1, self.length)[:, None, None, :]
        new = []
        for i, b in enumerate(self.blocks):
            h = b.n1(x); k1, v1 = b.self_att.kv(h)
            k = cache[2 * i] * (1 - put) + k1 * put; v = cache[2 * i + 1] * (1 - put) + v1 * put; new += [k, v]
            x, a = b(x, (cross[2 * i], cross[2 * i + 1]), self_kv=(k, v), vis=visible, training=training)
        return self.out_norm(x)[:, 0], (pos + 1,) + tuple(new), tf.reduce_mean(a, axis=1)[:, 0, :, None]

    def logits(self, out, training=False):
        return tf.matmul(out, self.embedding.embeddings, transpose_b=True)


def build_transformer(cfg, V):
    enc = TEncoder(cfg["d"], cfg["enc_layers"], cfg["heads"], cfg["ff"], cfg["dropout"], cfg["regions"])
    return enc, TDecoder(cfg["d"], cfg["dec_layers"], cfg["heads"], cfg["ff"], V, cfg["dropout"], cfg["T"] - 1)


# ---------------- CNN feature extractors (notebook Section 4), features rounded to float16 like the stored ones
def imagenet(name):
    # the local copy if present, otherwise Keras downloads the same file once (into ~/.keras/models)
    local = CNN_WEIGHTS / name
    return str(local) if local.exists() else "imagenet"


BACKBONES = {"inception": (lambda: keras.applications.InceptionV3(
                               include_top=False, weights=imagenet("inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")),
                           keras.applications.inception_v3.preprocess_input),
             "convnext_xl": (lambda: keras.applications.ConvNeXtXLarge(
                                 include_top=False, weights=imagenet("convnext_xlarge_notop.h5")), lambda x: x)}


class Captioner:
    def __init__(self, folder):
        self.cfg = cfg = json.loads((folder / "config.json").read_text())
        self.words = json.loads((folder / "vocab.json").read_text())
        self.kind = cfg.get("model", "transformer")
        self.enc, self.dec = (build_gru if self.kind == "gru" else build_transformer)(cfg, len(self.words))
        f0 = self.enc(tf.zeros((1, cfg["regions"], cfg["features"])))
        o0, *_ = self.dec.step(tf.zeros((1,), tf.int32), f0, self.dec.prepare(f0), self.dec.initial_state(1))
        self.dec.logits(o0); self.dec.built = True                    # all weights exist before they are loaded
        self.enc.load_weights(str(folder / "encoder.weights.h5")); self.dec.load_weights(str(folder / "decoder.weights.h5"))
        make, prep = BACKBONES[cfg["backbone"]]; base = make(); R, D, size = cfg["regions"], cfg["features"], cfg["image_size"]
        net = tf.function(lambda x: tf.cast(tf.cast(tf.reshape(base(prep(x), training=False), (-1, R, D)), tf.float16), tf.float32),
                          input_signature=[tf.TensorSpec((1, size, size, 3), tf.float32)])     # one trace for every image size
        self.features = lambda x: net(tf.image.resize(x, (size, size))[None])
        self.decoder_step = tf.function(self._decoder_step, reduce_retracing=True)

    def _decoder_step(self, word, feats, prep, state):
        out, state, a = self.dec.step(word, feats, prep, state)
        return tf.nn.log_softmax(self.dec.logits(out)), state

    def __call__(self, img):
        """img: float32 (H, W, 3) in [0, 1] -> caption (beam search of the notebook, Section 10)."""
        T, K = self.cfg["T"], self.cfg["beam_width"]; power = self.cfg["length_power"]
        feats = tf.repeat(self.enc(self.features(tf.constant(img * 255.0))), K, axis=0); prep = self.dec.prepare(feats)
        state = self.dec.initial_state(K)
        scores = np.full(K, -np.inf); scores[0] = 0.0
        seqs = np.zeros((K, T - 1), np.int32); done = np.zeros(K, bool); last = np.full(K, START, np.int32)
        for t in range(T - 1):
            lp, state = self.decoder_step(tf.constant(last), feats, prep, state)
            lp = lp.numpy(); lp[:, [PAD, UNK, START]] = -np.inf
            lp[done] = -np.inf; lp[done, PAD] = 0.0                           # a finished beam keeps its score
            cand = (scores[:, None] + lp).ravel(); top = np.argsort(-cand)[:K]
            scores = cand[top]; beam, word = top // lp.shape[1], (top % lp.shape[1]).astype(np.int32)
            seqs = seqs[beam]; seqs[:, t] = word
            done = done[beam] | (word == END) | (word == PAD)
            state = tuple(tf.gather(s, beam) for s in state); last = word
            if done.all():
                break
        lengths = np.maximum(((seqs != PAD) & (seqs != END)).sum(-1), 1)
        row = seqs[(scores / (((5 + lengths) / 6.0) ** power)).argmax()]
        out = []
        for i in row:
            if i in (END, PAD):
                break
            out.append(self.words[i])
        return " ".join(out)


# ---------------- Deliverables 1 and 2 (notebook Section 12)
classifier = keras.models.load_model(D1_MODEL, compile=False)
denoiser = keras.models.load_model(D2_MODEL, compile=False)
TILE_BATCH = 8
_denoise_tiles = tf.function(lambda x: denoiser(x, training=False),
                             input_signature=[tf.TensorSpec((TILE_BATCH, TILE + 2 * MARGIN, TILE + 2 * MARGIN, 3), tf.float32)])
_classify = tf.function(lambda x: classifier(x, training=False),
                        input_signature=[tf.TensorSpec((1,) + tuple(classifier.input_shape[1:]), tf.float32)])
captioners = {name: Captioner(folder) for name, folder in CAPTIONERS.items()}


def read_image(data):
    """File bytes -> float32 RGB in [0, 1], camera rotation applied, longest side limited to MAX_SIDE."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def denoise(img, batch=TILE_BATCH):
    # Deliverable 2 inference: 256 x 256 tiles with 64 pixels of reflected context
    h, w = img.shape[:2]; H, W_ = h + -h % TILE, w + -w % TILE
    x = np.pad(img, ((MARGIN, MARGIN + H - h), (MARGIN, MARGIN + W_ - w), (0, 0)), mode="reflect")
    corners = [(i, j) for i in range(0, H, TILE) for j in range(0, W_, TILE)]
    out = np.empty((H, W_, 3), np.float32)
    for s in range(0, len(corners), batch):
        part = corners[s:s + batch]
        tiles = np.stack([x[i:i + TILE + 2 * MARGIN, j:j + TILE + 2 * MARGIN] for i, j in part])
        tiles = np.pad(tiles, ((0, batch - len(part)), (0, 0), (0, 0), (0, 0)))     # always full batches: one trace
        y = _denoise_tiles(tf.constant(tiles)).numpy()
        for (i, j), t in zip(part, y):
            out[i:i + TILE, j:j + TILE] = t[MARGIN:MARGIN + TILE, MARGIN:MARGIN + TILE]
    return np.clip(out[:h, :w], 0.0, 1.0)


def photo_probability(img):
    # Deliverable 1 preparation: bilinear resize to 180 x 180, values 0-255
    x = tf.image.resize(tf.constant(img * 255.0), classifier.input_shape[1:3], method="bilinear")
    return float(_classify(x[None])[0, 0])


def run(data, denoise_first=DENOISE_FIRST):
    """One image file -> input, denoised image, P(photo), decision and the caption of each captioner."""
    img = read_image(data)
    if denoise_first:
        cleaned = denoise(img); p = photo_probability(cleaned)
    else:
        p = photo_probability(img); cleaned = denoise(img) if p >= PHOTO_THRESHOLD else None
    photo = p >= PHOTO_THRESHOLD
    return {"input": img, "denoised": cleaned, "p_photo": p, "photo": photo,
            "captions": {name: c(cleaned) if photo else None for name, c in captioners.items()}}


def warm_up():
    """Traces every function once so that the first uploaded image is not slow."""
    img = np.random.default_rng(0).random((240, 320, 3)).astype(np.float32)
    denoise(img); photo_probability(img)
    for c in captioners.values():
        c(img)
