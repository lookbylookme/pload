import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222", "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "worker hosts")
tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_workers", 2, "Number of workers")
tf.app.flags.DEFINE_boolean("is_sync", False, "using synchronous training or not")

FLAGS = tf.app.flags.FLAGS


def model(images,train=True):

    image_size = 28
    num_channels = 1
    num_categories = 10
    num_filters = 32
    filter_size = 5
    num_epochs = 2000
    keep_prob = 0.6
    """Define a simple mnist classifier"""
    conv1 = tf.layers.conv2d(images, num_filters, filter_size, padding='same', activation=tf.nn.relu)
    drop1 = tf.layers.dropout(conv1, keep_prob, training=train)
    pool1 = tf.layers.max_pooling2d(drop1, 2, 2)
    conv2 = tf.layers.conv2d(pool1, num_filters, filter_size, padding='same', activation=tf.nn.relu)
    drop2 = tf.layers.dropout(conv2, keep_prob, training=train)
    pool2 = tf.layers.max_pooling2d(drop2, 2, 2)
    conv3 = tf.layers.conv2d(pool2, num_filters, filter_size, padding='same', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2)
    conv4 = tf.layers.conv2d(pool3, num_filters, filter_size, padding='same', activation=tf.nn.relu)
    drop3 = tf.layers.dropout(conv4, keep_prob, training=train)
    flatten = tf.reshape(drop3, [-1, 9])
    with tf.contrib.framework.arg_scope(
    [tf.contrib.layers.fully_connected],
    normalizer_fn=tf.contrib.layers.batch_norm,
    normalizer_params={'is_training': train}):
        fc1 = tf.contrib.layers.fully_connected(flatten, 512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_categories, activation_fn=None)
    return fc2



def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # create the cluster configured by `ps_hosts' and 'worker_hosts'
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # create a server for local task
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()  # ps hosts only join
    elif FLAGS.job_name == "worker":
        # workers perform the operation
        # ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(FLAGS.num_ps)

        # Note: tf.train.replica_device_setter automatically place the paramters (Variables)
        # on the ps hosts (default placement strategy:  round-robin over all ps hosts, and also
        # place multi copies of operations to each worker host
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % (FLAGS.task_index),
                                                      cluster=cluster)):
            # load mnist dataset
            mnist = read_data_sets("./dataset", one_hot=True)

            # the model
            # images = tf.placeholder(tf.float32, [None, 784])
            images = tf.placeholder(tf.float32, [None, 28, 28, 1])
            labels = tf.placeholder(tf.int32, [None, 10])

            image_size = 28
            num_channels = 1
            num_categories = 10
            num_filters = 32
            filter_size = 5
            num_epochs = 2000
            keep_prob = 0.6

            train=True
            """Define a simple mnist classifier"""
            conv1 = tf.layers.conv2d(images, num_filters, filter_size, padding='same', activation=tf.nn.relu)
            drop1 = tf.layers.dropout(conv1, keep_prob, training=train)
            pool1 = tf.layers.max_pooling2d(drop1, 2, 2)
            conv2 = tf.layers.conv2d(pool1, num_filters, filter_size, padding='same', activation=tf.nn.relu)
            drop2 = tf.layers.dropout(conv2, keep_prob, training=train)
            pool2 = tf.layers.max_pooling2d(drop2, 2, 2)
            conv3 = tf.layers.conv2d(pool2, num_filters, filter_size, padding='same', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(conv3, 2, 2)
            conv4 = tf.layers.conv2d(pool3, num_filters, filter_size, padding='same', activation=tf.nn.relu)
            drop3 = tf.layers.dropout(conv4, keep_prob, training=train)
            flatten = tf.reshape(drop3, [-1, 288])
            with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.fully_connected],
            normalizer_fn=tf.contrib.layers.batch_norm,
            normalizer_params={'is_training': train}):
                fc1 = tf.contrib.layers.fully_connected(flatten, 512)
                fc2 = tf.contrib.layers.fully_connected(fc1, num_categories, activation_fn=None)


            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=labels))

            # The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=2000)]

            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-04)

            if FLAGS.is_sync:
                # asynchronous training
                # use tf.train.SyncReplicasOptimizer wrap optimizer
                # ref: https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
                optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=FLAGS.num_workers,
                                                       total_num_replicas=FLAGS.num_workers)
                # create the hook which handles initialization and queues
                hooks.append(optimizer.make_session_run_hook((FLAGS.task_index==0)))

            train_op = optimizer.minimize(loss, global_step=global_step,
                                          aggregation_method=tf.AggregationMethod.ADD_N)

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   checkpoint_dir=None,
                                                   hooks=hooks) as mon_sess:

                while not mon_sess.should_stop():
                    # mon_sess.run handles AbortedError in case of preempted PS.
                    img_batch, label_batch = mnist.train.next_batch(32)
                    img_batch = img_batch.reshape((-1,28,28,1))

                    _, ls, step = mon_sess.run([train_op, loss, global_step],
                                               feed_dict={images: img_batch, labels: label_batch})
                    if step % 100 == 0:
                        print("Train step %d, loss: %f" % (step, ls))

if __name__ == "__main__":
    tf.app.run()
