"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()

batch_size = 128*4

#%% Sentence to Sequence
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    sentence_ids = [vocab_to_int.get(word.lower(), vocab_to_int['<UNK>']) for word in sentence.split()]
    
    return sentence_ids


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)

#%% Translate
sentences = []
sentences.append('he saw a old yellow truck .')
sentences.append('the wonderful shark thinks the field is green and freezing')
sentences.append('the strawberry was liked , but it is warm .')
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
for sentence in sentences:
    translate_sentence = sentence_to_seq(sentence, source_vocab_to_int)
    
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)
    
        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
        translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                             target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                             source_sequence_length: [len(translate_sentence)]*batch_size,
                                             keep_prob: 1.0})[0]
    
    print('Input')
    print('  Word Ids:      {}'.format([i for i in translate_sentence]))
    print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))
    
    print('\nPrediction')
    print('  Word Ids:      {}'.format([i for i in translate_logits]))
    print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))



