import datetime
import time
import os
import sys
import re
from rat import ratdb

import settings


def write_to_log(logfile, message):
   ''' 
   This is a helper-function to write to the logfile
   '''
   now = datetime.datetime.now().ctime()
   write_string = '%s : %s \n' % (now, message)
   logfile.write(write_string)


def get_run_number_fromfile(file):
   '''
   Retrieve run number from file name
   '''
   zdab_regex = re.compile(r"^SNO?\w_0*(\d+)_\d+\.l2.zdab$")
   matches = zdab_regex.match(file)
   if matches:
      return int(matches.group(1))
   else:
      return 0

def get_macro_from_file(file):
   '''
   Retrieve run number from file name
   '''
   regmacro = re.compile(r"^SNO?\w_0*(\d+)_\d+\.l2.zdab$")
   matches = zdab_regex.match(file)
   if matches:
      return int(matches.group(1))
   else:
      return 0

class cd:
   '''
   Context manager for changing the current working directory
   '''
   def __init__(self, newPath):
      self.newPath = newPath

   def __enter__(self):
      self.savedPath = os.getcwd()
      os.chdir(self.newPath)

   def __exit__(self, etype, value, traceback):
      os.chdir(self.savedPath)


def write_to_ratdb(filename, logfile):
    '''
    Pushes file to RATDB with the ratdb tool and 
    return whether it was successful
    ''' 
    attempt=0
    ratdbLink = None
    in_ratdb=False
    while not in_ratdb and attempt < 5:
        try:
            ratdbLink = ratdb.RATDBConnector('postgres://'+settings.RATDBWRITE_AUTH+'@'+settings.RATDB_SERVER+'/ratdb')
            in_ratdb = len( ratdbLink.upload([filename]) )
        except Exception, error:
            print ("ratdb error %s \n" % str(error))
            write_to_log(logfile, 'Could NOT upload to RATDB due to the error: %s. \n' % error)
            write_to_log(logfile, 'Will retry in 10s... .\n')
            attempt += 1
            if ratdbLink is not None:
                ratdbLink.close_ratdb_connection()
            time.sleep(10.)
    return in_ratdb


def is_table_in_ratdb(run_number, logfile, table_name):
    '''
    Check whether a table exists in ratdb.
    '''
    try:
        ratdbLink = ratdb.RATDBConnector('postgres://'+settings.RATDBWRITE_AUTH+'@'+settings.RATDB_SERVER+'/ratdb')
        table = ratdbLink.fetch(obj_type=table_name,run=run_number)
        return len(table)
    except:
        write_to_log(logfile, " Error: Request table: %s not in ratdb for run %i" % (table_name, run_number))
        return False

