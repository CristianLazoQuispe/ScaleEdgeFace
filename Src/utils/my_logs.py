
import logging
import datetime

import sys

DATE_FORMAT = '%Y-%m-%d' 


class Logger:
    def __init__(self, tag, out_type=sys.stdout):
        self.tag = tag
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        #handler = logging.FileHandler('results/logs/log_' + strftime("%Y%m%d", gmtime()) + '.txt', mode='a')
        handler = logging.FileHandler('results/logs/log_' + datetime.datetime.today().strftime(DATE_FORMAT) + '.txt', mode='a')
        handler.setFormatter(formatter)
        screen_handler = logging.StreamHandler(stream=out_type)
        screen_handler.setFormatter(formatter)
        logger = logging.getLogger(tag)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(screen_handler)

        self.log = logger
    def reinit(self, out_type=sys.stdout):
        while self.log.hasHandlers():
            self.log.removeHandler(self.log.handlers[0])
            #print('deleting handler')
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        #handler = logging.FileHandler('results/logs/log_' + strftime("%Y%m%d", gmtime()) + '.txt', mode='a')
        handler = logging.FileHandler('results/logs/log_' + datetime.datetime.today().strftime(DATE_FORMAT) + '.txt', mode='a')
        handler.setFormatter(formatter)
        screen_handler = logging.StreamHandler(stream=out_type)
        screen_handler.setFormatter(formatter)
        logger = logging.getLogger(self.tag)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(screen_handler)
        self.log = logger
    def format_message(self, message):
        return "[Owner={0}] {1}".format("{:<12}".format(self.tag), message)

    def error(self, message):
        self.log.error(self.format_message(message))

    def info(self, message):
        self.log.info(self.format_message(message))

    def warning(self, message):
        self.log.warning(self.format_message(message))

    def debug(self, message):
        self.log.debug(self.format_message(message))

    def critical(self, message):
        self.log.critical(self.format_message(message))