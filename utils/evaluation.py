from tqdm import tqdm
import tensorflow as tf
import numpy as np

from models.transformer.utils import CustomSchedule, create_masks
from models.transformer import Transformer
from utils.data import postprocessing, checkout_data
from utils.dataloader import encoder_preprocess, decoder_preprocess


def translate(inputfile, pred_file_path,
              num_layers, d_model, num_heads, dff,
              enc_data="data/aligned_unformated_en",
              dec_data="data/aligned_formated_fr",
              dropout_rate=0.3, batch_size=32):
    _, encoder_v2id, _ = encoder_preprocess(data=enc_data)
    _, decoder_v2id, _ = decoder_preprocess(data=dec_data)
    input_vocab_size = len(encoder_v2id) + 1
    target_vocab_size = len(decoder_v2id) + 1

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    checkpoint_path = "./checkpoints/train/w2w/unformated_en_2_formated_fr"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=50
    )

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

    def evaluate(encoder_input, batch_size, k=2):
        batch_size = encoder_input.shape[0]
        max_length = encoder_input.shape[-1]
        queue = [
            (
                np.array([decoder_v2id['<SOS>']] * batch_size).reshape(-1, 1),
                np.ones((batch_size, 1))
            )
        ]

        for _ in range(max_length):
            new_queue = []
            while len(queue) > 0:
                candidate_batch, probs = queue.pop()

                (
                    enc_padding_mask, combined_mask, dec_padding_mask
                ) = create_masks(
                    encoder_input, candidate_batch
                )

                # predictions.shape == (batch_size, seq_len, vocab_size)
                (
                    predictions, attention_weights
                ) = transformer(encoder_input,
                                candidate_batch,
                                False,
                                enc_padding_mask,
                                combined_mask,
                                dec_padding_mask)

                # select the last word from the seq_len dimension
                # (batch_size, seq_len, vocab_size)
                predictions = tf.nn.softmax(predictions, axis=2)
                k_top_predictions = tf.argsort(predictions, axis=2)[:, -1, -k:]
                k_top_probs = tf.sort(predictions, axis=2)[:, -1, -k:]

                for j in range(k):
                    if j >= k_top_predictions.shape[-1]:
                        break

                    top_probs = k_top_probs[:, j]
                    top_preds = tf.expand_dims(k_top_predictions[:, j], 1)

                    new_queue.insert(
                        0,
                        (
                            tf.concat([candidate_batch, top_preds], axis=1),
                            probs * -1 * tf.math.log(top_probs)
                        )
                    )

            queue = sorted(new_queue, key=lambda tup: np.sum(tup[1]))[:k]

        output = queue[-1][0]
        return output, attention_weights

    def get_gen(data, batch_size):
        def data_word_generator():
            ch_data = checkout_data(data)
            size = len(ch_data)
            bs = min(size, batch_size)
            steps = size // bs
            init = 0
            end = bs
            for _ in range(steps):
                to_return = ch_data[init:end]
                init = end
                end += bs
                yield to_return
        return data_word_generator()

    enc_gen = get_gen(
        data=inputfile,
        batch_size=batch_size
    )

    print(pred_file_path)

    f = open(pred_file_path, "w")
    for enc_data_words in tqdm(enc_gen):
        enc_data_int, _, _ = encoder_preprocess(data=enc_data_words)
        out, _ = evaluate(enc_data_int, batch_size)
        out_words = postprocessing(
            dec_data=out,
            dec_v2id=decoder_v2id,
            Print=False,
            tokenize_type="w",
            fasttext_model="embeddings/unformated_en_w2w/%d/unaligned_unformated_en" %
            (d_model),
            enc_data=enc_data_words,
            remove_punctuation=True,
            lower=False,
            CAP=True,
            NUM=True,
            ALNUM=True,
            UPPER=True,
            enc_v2id=encoder_v2id)

        print('\n'.join(out_words), file=f)
    f.close()
