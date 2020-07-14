import argparse
import numpy as np
import tensorflow as tf
import os

from models.tf_albert_modeling import AlbertModelMRC, AlbertConfig
from optimizations.tf_optimization import Optimizer
import json
import utils
from evaluate.cmrc2018_evaluate import get_eval
from evaluate.cmrc2018_output import write_predictions
import random
from tqdm import tqdm
import collections
from tokenizations.offical_tokenization import BertTokenizer
from preprocess.cmrc2018_preprocess import json2features

def data_generator(data, n_batch, shuffle=False, drop_last=False):
    steps_per_epoch = len(data) // n_batch
    if len(data) % n_batch != 0 and not drop_last:
        steps_per_epoch += 1
    data_set = dict()
    for k in data[0]:
        data_set[k] = np.array([data_[k] for data_ in data])
    index_all = np.arange(len(data))

    while True:
        if shuffle:
            random.shuffle(index_all)
        for i in range(steps_per_epoch):
            yield {k: data_set[k][index_all[i * n_batch:(i + 1) * n_batch]] for k in data_set}
            
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser.add_argument('--gpu_ids', type=str, default='0')

    # training parameter
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--n_batch', type=int, default=24)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--loss_scale', type=float, default=2.0 ** 15)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--loss_count', type=int, default=1000)
    parser.add_argument('--seed', type=list, default=[123, 456, 789, 556, 977])
    parser.add_argument('--float16', type=int, default=False)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=30)  # show the average loss per 30 steps args.
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--eval_epochs', type=float, default=0.5)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--vocab_size', type=int, default=21128)
    parser.add_argument('--max_seq_length', type=int, default=512)

    # data dir
    parser.add_argument('--vocab_file', type=str,
                        default='albert_small_zh_google/vocab.txt')

    parser.add_argument('--eval_file', type=str, default='squad-style-data/cmrc2018_eval.json')
    parser.add_argument('--eval_dir1', type=str, default='dataset/cmrc2018/eval_examples_roberta512.json')
    parser.add_argument('--eval_dir2', type=str, default='dataset/cmrc2018/eval_features_roberta512.json')
    parser.add_argument('--bert_config_file', type=str,
                        default='albert_small_zh_google/albert_config_small_google.json')
    parser.add_argument('--init_restore_dir', type=str,
                        default='albert_small_zh_google/albert_model.ckpt')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='check_points/cmrc2018/albert/')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    # use some global vars for convenience
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    n_gpu = len(args.gpu_ids.split(','))
    
    args.checkpoint_dir += ('/epoch{}_batch{}_lr{}_warmup{}_anslen{}_tf/'
                            .format(args.train_epochs, args.n_batch, args.lr, args.warmup_rate, args.max_ans_length))
    mpi_rank = 0
    args = utils.check_args(args, mpi_rank)
    
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)

    if not os.path.exists(args.eval_dir1) or not os.path.exists(args.eval_dir2):
        json2features(args.eval_file, [args.eval_dir1, args.eval_dir2], tokenizer, is_training=False)

    
    eval_examples = json.load(open(args.eval_dir1, 'r'))
    eval_data = json.load(open(args.eval_dir2, 'r'))
    eval_steps_per_epoch = len(eval_data) // (args.n_batch * n_gpu)

    eval_gen = data_generator(eval_data, args.n_batch * n_gpu, shuffle=False, drop_last=False)

    if len(eval_data) % (args.n_batch * n_gpu) != 0:
        eval_steps_per_epoch += 1
        
    with tf.device("/gpu:0"):
        input_ids = tf.placeholder(tf.int32, shape=[None, args.max_seq_length], name='input_ids')
        input_masks = tf.placeholder(tf.float32, shape=[None, args.max_seq_length], name='input_masks')
        segment_ids = tf.placeholder(tf.int32, shape=[None, args.max_seq_length], name='segment_ids')
        start_positions = tf.placeholder(tf.int32, shape=[None, ], name='start_positions')
        end_positions = tf.placeholder(tf.int32, shape=[None, ], name='end_positions')
    
    configsession = tf.ConfigProto()
    configsession.gpu_options.allow_growth = True
    sess = tf.Session(config=configsession)
    
    RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits"])
    
    with sess.as_default():
        bert_config = AlbertConfig.from_json_file(args.bert_config_file)
        eval_model = AlbertModelMRC(config=bert_config,
                                is_training=False,
                                input_ids=input_ids,
                                input_mask=input_masks,
                                token_type_ids=segment_ids,
                                use_float16=args.float16)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())# 这里尤其注意，先初始化，在加载参数，否者会把bert的参数重新初始化。这里和demo1是有区别的
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))

#         for i in tf.get_default_graph().get_operations():
#             print(i.name)
    
        print('Evaluating...')
        all_results = []
        for i_step in tqdm(range(eval_steps_per_epoch),
                           disable=False if mpi_rank == 0 else True):
            batch_data = next(eval_gen)
            feed_data = {input_ids: batch_data['input_ids'],
                         input_masks: batch_data['input_mask'],
                         segment_ids: batch_data['segment_ids']}
            
            batch_start_logits, batch_end_logits = sess.run(
                [eval_model.start_logits, eval_model.end_logits],
                feed_dict=feed_data)
            
            for j in range(len(batch_data['unique_id'])):
                start_logits = batch_start_logits[j]
                end_logits = batch_end_logits[j]
                unique_id = batch_data['unique_id'][j]
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))
        output_prediction_file = os.path.join("./eval_results/",
                                                                      'prediction_epoch'  + '.json')
        output_nbest_file = os.path.join("./eval_results/", 'nbest_epoch' + '.json')

        write_predictions(eval_examples, eval_data, all_results,
                          n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                          do_lower_case=True, output_prediction_file=output_prediction_file,
                          output_nbest_file=output_nbest_file)


