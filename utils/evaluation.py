from tqdm import tqdm
import os
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
              epoch=28, dropout_rate=0.3, batch_size=32):
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
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    def evaluate(encoder_input):
        output = np.array([decoder_v2id['<SOS>']]*batch_size).reshape(-1, 1)
        MAX_LENGTH = encoder_input.shape[-1]

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            # if predicted_id == tokenizer_en.vocab_size+1:
            #   return output, attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=1)

        return output, attention_weights

    def get_gen(data, batch_size):
        def data_word_generator():
            ch_data = checkout_data(data)[int(-11000*0.2):]
            size = len(ch_data)
            steps = size//batch_size + 1
            init = 0
            end = batch_size
            for i in range(steps):
                to_return = ch_data[init:end]
                init = end
                end += batch_size
                yield to_return
        return data_word_generator()

    enc_gen = get_gen(
        data=inputfile,
        batch_size=batch_size
    )

    f = open(pred_file_path, "w")
    for enc_data_words in tqdm(enc_gen):
        enc_data_int, _, _ = encoder_preprocess(data=enc_data_words)
        out, _ = evaluate(enc_data_int)
        out_words = postprocessing(
            dec_data=out,
            dec_v2id=decoder_v2id,
            Print=False,
            tokenize_type="w",
            fasttext_model="embeddings/unformated_en_w2w/%d/unaligned_unformated_en" % (d_model),
            enc_data=enc_data_words,
            remove_punctuation=True,
            lower=False,
            CAP=True,
            NUM=True,
            ALNUM=True,
            UPPER=True,
            enc_v2id=encoder_v2id
        )
        f.write(' '.join(out_words))
    f.close()
