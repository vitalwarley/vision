"""Scheduler for learning rate."""

def scheduler(epoch, lr):
    # return lr * tf.math.exp(-0.1)
    # return lr * 1.005
    return lr / (1 + epoch)  # learning rate decay

