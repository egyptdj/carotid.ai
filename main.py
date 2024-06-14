import os
import csv
import json
import glob
import random
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from delong import get_auc_ci
from matplotlib.ticker import FormatStrFormatter


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

YAMNET_HANDLE = 'https://tfhub.dev/google/yamnet/1'
SAMPLE_RATE = 16000
INPUT_SECONDS = None
INPUT_LENGTH = 15600
BATCH_SIZE = 1
SEED = 0

if INPUT_SECONDS: INPUT_LENGTH = int(SAMPLE_RATE * INPUT_SECONDS)

MODEL_NAME = 'model'
BASEDIR = 'path_to_basedir'
DATADIR = os.path.join(BASEDIR, 'data')
MODELDIR = os.path.join(BASEDIR, 'models', f'{MODEL_NAME}_{INPUT_LENGTH}')
os.makedirs(MODELDIR, exist_ok=True)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    train_audio_file_list = pd.Series(glob.glob(os.path.join(DATADIR, 'train', '**/*.wav')), dtype='object')
    train_audio_label_list = [f.split('/')[-2] for f in train_audio_file_list]


    val_audio_file_list = pd.Series(glob.glob(os.path.join(DATADIR, 'val', '**/*.wav')), dtype='object')
    val_audio_label_list = [f.split('/')[-2] for f in val_audio_file_list]


    test_audio_file_list = pd.Series(glob.glob(os.path.join(DATADIR, 'test', '**/*.wav')), dtype='object')
    test_audio_label_list = [f.split('/')[-2] for f in test_audio_file_list]

    class_name_dict = {}
    i = 0
    for label in train_audio_label_list:
        if not label in list(class_name_dict.keys()):
            class_name_dict[label] = i
            i += 1

    with open(os.path.join(MODELDIR, 'class_name_dict.json'), 'w') as f:
        json.dump(class_name_dict, f)
    
    train_audio_label_list = pd.Series([class_name_dict[i] for i in train_audio_label_list], dtype='int64')
    val_audio_label_list = pd.Series([class_name_dict[i] for i in val_audio_label_list], dtype='int64')
    test_audio_label_list = pd.Series([class_name_dict[i] for i in test_audio_label_list], dtype='int64')


    train_audio_label_list_step1 = train_audio_label_list.map(lambda x: 0 if x==class_name_dict['background'] else 1)
    val_audio_label_list_step1 = val_audio_label_list.map(lambda x: 0 if x==class_name_dict['background'] else 1)
    test_audio_label_list_step1 = test_audio_label_list.map(lambda x: 0 if x==class_name_dict['background'] else 1)


    train_audio_label_list_step2 = train_audio_label_list.replace({class_name_dict['background']: -1, class_name_dict['normal']: 0, class_name_dict['bruit']: 1, class_name_dict['normal-noise']: 0, class_name_dict['bruit-noise']: 0})
    val_audio_label_list_step2 = val_audio_label_list.replace({class_name_dict['background']: -1, class_name_dict['normal']: 0, class_name_dict['bruit']: 1, class_name_dict['normal-noise']: 0, class_name_dict['bruit-noise']: 0})
    test_audio_label_list_step2 = test_audio_label_list.replace({class_name_dict['background']: -1, class_name_dict['normal']: 0, class_name_dict['bruit']: 1, class_name_dict['normal-noise']: 0, class_name_dict['bruit-noise']: 0})


    train_ds1 = tf.data.Dataset.from_tensor_slices((train_audio_file_list, train_audio_label_list_step1))
    val_ds1 = tf.data.Dataset.from_tensor_slices((val_audio_file_list, val_audio_label_list_step1))
    test_ds1 = tf.data.Dataset.from_tensor_slices((test_audio_file_list, test_audio_label_list_step1))


    train_ds2 = tf.data.Dataset.from_tensor_slices((train_audio_file_list, train_audio_label_list_step2)).filter(lambda x, y: tf.math.logical_not(tf.math.equal(y, -1)))
    val_ds2 = tf.data.Dataset.from_tensor_slices((val_audio_file_list, val_audio_label_list_step2)).filter(lambda x, y: tf.math.logical_not(tf.math.equal(y, -1)))
    test_ds2 = tf.data.Dataset.from_tensor_slices((test_audio_file_list, test_audio_label_list_step2)).filter(lambda x, y: tf.math.logical_not(tf.math.equal(y, -1)))
    
    @tf.function
    def load_wav_and_label(audio_file, label):
        audio = tf.io.read_file(audio_file)
        audio, _ = tf.audio.decode_wav(audio)
        audio = tf.squeeze(audio, axis=-1)
        return audio, label

    yamnet_model = hub.load(YAMNET_HANDLE)

    @tf.function
    def extract_embedding(wav_data, label):
        scores, embeddings, spectrogram = yamnet_model(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        return embeddings, tf.repeat(label, num_embeddings)


    model_step1 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='yamnet_embedding_step1'),
            tf.keras.layers.Dense(1
                , activation='sigmoid'
            ),
    ])

    model_step2 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='yamnet_embedding_step2'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1
                , activation='sigmoid'                                            
            ),
    ])

    model_step1.summary()
    model_step2.summary()

    metrics = ['accuracy', 
                        tf.keras.metrics.AUC(), 
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall()
                        ]

    model_step1.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5,
        ),
        metrics=metrics,
    )


    model_step2.compile(
        # loss=tf.keras.losses.CategoricalCrossentropy(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5,
        ),
        metrics=metrics,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
        ),
        tf.keras.callbacks.LearningRateScheduler(
            tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-5,
                decay_steps=100,
            )
        )
    ]

    callbacks2 = [
        tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
        ),
        tf.keras.callbacks.LearningRateScheduler(
            tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-4,
                decay_steps=100,
            )
        )
    ]

    train_ds1 = train_ds1.map(load_wav_and_label).map(extract_embedding).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds1 = val_ds1.map(load_wav_and_label).map(extract_embedding).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds1 = test_ds1.map(load_wav_and_label).map(extract_embedding).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds2 = train_ds2.map(load_wav_and_label).map(extract_embedding).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds2 = val_ds2.map(load_wav_and_label).map(extract_embedding).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds2 = test_ds2.map(load_wav_and_label).map(extract_embedding).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    history1 = model_step1.fit(train_ds1,
                                            epochs=100,
                                            validation_data=val_ds1,
                                            callbacks=callbacks
                                            )

    test_history1 = model_step1.evaluate(test_ds1, return_dict=True)
    y_pred1 = np.concatenate(model_step1.predict(test_ds1).numpy()).squeeze()
    y_true1 = []
    for x, y in test_ds1:
        y_true1.append(y.numpy())
    y_true1 = np.concatenate(y_true1, axis=1).squeeze()

    history2 = model_step2.fit(train_ds2,
                                            epochs=100,
                                            validation_data=val_ds2,
                                            callbacks=callbacks2
                                            )

    test_history2 = model_step2.evaluate(test_ds2, return_dict=True)
    y_pred2 = np.concatenate(model_step2.predict(test_ds2).numpy()).squeeze()
    y_true2 = []
    for x, y in test_ds2:
        y_true2.append(y.numpy())
    y_true2 = np.concatenate(y_true2, axis=1).squeeze()
    
    auc1, auc1cov, auc1ci = get_auc_ci(y_true1, y_pred1)
    auc2, auc2cov, auc2ci = get_auc_ci(y_true2, y_pred2)
    
    test_history1['aucmean'] = auc1
    test_history1['auccov'] = auc1cov
    test_history1['aucci_min'] = auc1ci[0]
    test_history1['aucci_max'] = auc1ci[1]

    with open(os.path.join(MODELDIR, 'test_metrics_step1.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerows(test_history1.items())
        
    test_history2['aucmean'] = auc2
    test_history2['auccov'] = auc2cov
    test_history2['aucci_min'] = auc2ci[0]
    test_history2['aucci_max'] = auc2ci[1]

    with open(os.path.join(MODELDIR, 'test_metrics_step2.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerows(test_history2.items())

    embedding_extraction_layer = hub.KerasLayer(YAMNET_HANDLE, input_shape=(INPUT_LENGTH,), trainable=False, name='yamnet')
    
    train_history1 = pd.DataFrame.from_dict({
        'Loss': history1.history['loss'], 
        'Accuracy': history1.history['accuracy'], 
        'Precision': history1.history['precision'], 
        'Recall': history1.history['recall'],
        'AUC': history1.history['auc'],
        'Split': ['Train' for _ in history1.history['loss']],
        'Epoch': [i+1 for i, _ in enumerate(history1.history['loss'])],
    })
    val_history1 = pd.DataFrame.from_dict({
        'Loss': history1.history['val_loss'], 
        'Accuracy': history1.history['val_accuracy'], 
        'Precision': history1.history['val_precision'], 
        'Recall': history1.history['val_recall'], 
        'AUC': history1.history['val_auc'], 
        'Split': ['Validation' for _ in history1.history['val_loss']],
        'Epoch': [i+1 for i, _ in enumerate(history1.history['val_loss'])],
    })
    history1_df = pd.concat([train_history1, val_history1])

    train_history2 = pd.DataFrame.from_dict({
        'Loss': history2.history['loss'], 
        'Accuracy': history2.history['accuracy'], 
        'Precision': history2.history['precision'], 
        'Recall': history2.history['recall'],
        'AUC': history2.history['auc'],
        'Split': ['Train' for _ in history2.history['loss']],
        'Epoch': [i+1 for i, _ in enumerate(history2.history['loss'])],
    })
    val_history2 = pd.DataFrame.from_dict({
        'Loss': history2.history['val_loss'], 
        'Accuracy': history2.history['val_accuracy'], 
        'Precision': history2.history['val_precision'], 
        'Recall': history2.history['val_recall'], 
        'AUC': history2.history['val_auc'], 
        'Split': ['Validation' for _ in history2.history['val_loss']],
        'Epoch': [i+1 for i, _ in enumerate(history2.history['val_loss'])],
    })
    history2_df = pd.concat([train_history2, val_history2])
    
    sns.set_theme(context='paper', style='whitegrid', font='helvetica', font_scale=1.5, palette='muted')

    fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(14,7), sharex=True)
    for i, metric in enumerate(['Loss', 'Accuracy', 'Precision', 'Recall', 'AUC']):
        h2 = sns.lineplot(history2_df, x='Epoch', y=metric, hue='Split', linewidth=2, legend=False, ax=ax[0][i])
        h1 = sns.lineplot(history1_df, x='Epoch', y=metric, hue='Split', linewidth=2, legend=True if i==4 else False, ax=ax[1][i])
        h2.set(ylabel='Bruit classifier' if i==0 else None, title=metric)
        h1.set(ylabel='Carotic sound recognizer' if i==0 else None, xlabel='Epoch' if i==2 else None)
        h2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        h1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.suptitle('Traning curve')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELDIR, 'traincurve.png'))
    plt.close()
    
    model_step1.layers[-1].activation = None
    model_step2.layers[-1].activation = None

    print(model_step1.summary())
    print(model_step2.summary())
    
    model_step1.compile()
    model_step2.compile()
    
    input_segment = tf.keras.layers.Input(batch_size=1, shape=(INPUT_LENGTH, ), dtype=tf.float32, name='audio')
    input_segment_reshaped = tf.reshape(input_segment, (INPUT_LENGTH,), name='reshape')
    _, embeddings, _ = embedding_extraction_layer(input_segment_reshaped)
    step1_output = tf.sigmoid(tf.math.reduce_mean(model_step1(embeddings), axis=0))
    step2_output = tf.math.reduce_mean(model_step2(embeddings), axis=0)
    step2_output = tf.nn.sigmoid(step2_output)
    final_output = tf.concat([step1_output, step2_output], axis=0)
    serving_model = tf.keras.Model(input_segment, final_output, name='carotid_ai')
    serving_model.save(MODELDIR, include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_saved_model(MODELDIR)

    tflite_model = converter.convert()

    with open(os.path.join(MODELDIR, 'carotid_ai.tflite'), 'wb') as f:
        f.write(tflite_model)
        

if __name__ == '__main__':
    main()