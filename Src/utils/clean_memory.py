import threading
import logging
import os
import gc

def cerrar_hilos():
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()

def cerrar_archivos():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

def limpiar_gpu():
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except:
        None
def limpiar_ram():
    gc.collect()

def clean_memory():
    print("Cleaning memory..")
    cerrar_hilos()
    limpiar_gpu()
    cerrar_archivos()
    limpiar_ram()
