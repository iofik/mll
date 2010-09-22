import os
import re
import sys
import time
import codecs
import logging
import subprocess
from optparse import OptionParser
from datetime import datetime

import PoligonService as poligon

logger = logging

LEARN_INDEXES_FILE = "learn_indexes.txt"
TEST_INDEXES_FILE = "test_indexes.txt"
PARAMETERS_FILE = "parameters.txt"
PENALTIES_FILE = "penalties.txt"

LEARN_TARGETS_FILE = "learn_targets.txt"
LEARN_PROBABILITY_MATRIX_FILE = "learn_probabilities.txt"
LEARN_OBJECTS_WEIGHTS_FILE = "learn_objects_weights.txt"
LEARN_PROPERTIES_WEIGHTS_FILE = "learn_properties_weights.txt"

TEST_TARGETS_FILE = "test_targets.txt"
TEST_PROBABILITY_MATRIX_FILE = "test_probabilities.txt"
TEST_OBJECTS_WEIGHTS_FILE = "test_objects_weights.txt"
TEST_PROPERTIES_WEIGHTS_FILE = "test_properties_weights.txt"

def _save_as_arff(problem, synonim, path):
  """Writes problem data in arff format"""
  classes = []
  
  file = open(path, 'w')
  file.write('%\n%\n%\n\n')  
  file.write('@RELATION {0}\n\n'.format(synonim))
  
  attributes = []
  width = len(problem.PropertiesDescription.PropertyDescription)
  
  for i in range(width-1):
    nominals = {}
    property = problem.PropertiesDescription.PropertyDescription[i]   
    attribute = '@ATTRIBUTE A{0}\t'.format(i+1)
    if property.Type == 'Nominal':
      attribute = attribute + '{'
      for j in range(len(property.Values.Int)):
        tag = 'A{0}{1}'.format(i+1,j+1)
        nominals[int(property.Values.Int[j])] = tag
        attribute = attribute + '{0},'.format(tag)
      attribute = attribute.rstrip(',') + '}'
    else:
      attribute = attribute + 'REAL'
    attribute = attribute + '\n'
    file.write(attribute)
    attributes.append(nominals)
  
  attribute = '@ATTRIBUTE class\t{'  
  property = problem.PropertiesDescription.PropertyDescription[width-1]
  for value in property.Values.Int:
    classes.append(value)
    attribute = attribute + '{0},'.format(value)
  attribute = attribute.rstrip(',') + '}'
  attribute = attribute + '\n'  
  file.write(attribute)
    
  file.write('\n@DATA\n')
  
  for i in range(len(problem.DataMatrix.ArrayOfDouble)):
    row = ''
    for j in range(len(problem.DataMatrix.ArrayOfDouble[i].Double)):
      data = str(problem.DataMatrix.ArrayOfDouble[i].Double[j])
      if data.lower() == 'nan':
        data = '?'
      elif attributes[j]:
        data = attributes[j][int(problem.DataMatrix.ArrayOfDouble[i].Double[j])]
      row = row + '{0},'.format(data)
    row = row + '{0}\n'.format(problem.Target.Int[i])
    file.write(row)
    
  file.write('%\n%\n%\n\n')
  file.close()
  
  return len(problem.DataMatrix.ArrayOfDouble) \
    , len(problem.PropertiesDescription.PropertyDescription) \
    , classes
  
  
def _save_params(task, path):
  """Writes alg params into the given file"""
  # NOTE: parameter name and value shouldn't contain any blanks
  # TODO: use json format
  count = 0
  file = open(path, 'w')
  try:
    if task.AlgParamNames:
      for i in xrange(len(task.AlgParamNames.String)):
        if task.AlgParamUsages.Boolean[i]:
          count += 1
          file.write('{0}={1}\n'.format(
                      task.AlgParamNames.String[i]
                      , task.AlgParamValues.String[i]))
  except:
    logger.error('Error occured while saving params\n{0}'.format(sys.exc_info()[0]))
  file.write('\n')
  file.close()
  return count

def _load_vector(name, type=str):
  try:
    vector = []
    file = open(name)
    text = file.read()
    file.close()
    lines = text.split('\n')
    for line in lines:
      if (len(line)):
        vector.append(type(line))
    return vector
  except:
    return []
  
def _load_matrix(name, type=str):
  try:
    matrix = []
    file = open(name)
    text = file.read()
    file.close()
    lines = text.split('\n')
    for line in lines:
      if not line: break
      items = line.split()
      if not items: break
      vector = []
      for item in items:
        vector.append(type(item))
      matrix.append(vector)
    return matrix
  except:
    return []
  
def _save_vector(vector, filename):
  output = open(filename, 'w')
  for v in vector:
    print >> output, v
  output.close()
  
def _save_matrix(matrix, filename):
  output = open(filename, 'w')
  for line in matrix:
    for v in line:
      print >> output, v,
    print >> output
  output.close()

def _init_logger():
  today = datetime.today()
  logger = logging.getLogger('Poligon')
  logger.setLevel(logging.DEBUG)

  stream_handler = logging.StreamHandler()

  file_handler   = logging.FileHandler(
                    today.strftime('poligon_%Y%m%d_%H%M%S.log')
                    , encoding='utf-8')

  formatter = logging.Formatter(
                    "%(asctime)s\t%(levelname)s\t%(message)s")

  stream_handler.setFormatter(formatter)
  file_handler.setFormatter(formatter)

  logger.addHandler(stream_handler)
  logger.addHandler(file_handler)

  poligon.logger = logger

  return logger

def _init_optparser():
  parser = OptionParser()
  parser.add_option("-c", "--commands", dest="commands", default="commands.txt",
                    help="Algorithm commands file")
  parser.add_option("-s", "--algsynonim", dest="algsynonim",
                    help="Algorithm synonim at Poligon")
  parser.add_option("-p", "--algpassword", dest="algpassword",
                    help="Algorithm password at Poligon")
  parser.add_option("-d", "--datasets", dest="datasets", default="datasets",
                    help="Folder with saved datasets")
  parser.add_option("-t", "--timeout", dest="timeout", type="int", default=10,
                    help="Timeout for algorithm process, minutes")
  parser.add_option("-w", "--wait", dest="wait", type="int", default=0,
                    help="Wait period between series of requests, minutes (0 for requsting once)")
  return parser

class Result(object):

  class Data(object):

    def __init__(self):
      self.Error = False
      self.ErrorException = ''
      self.ProbabilityMatrix = []
      self.Targets = []
      self.PropertiesWeights = []
      self.ObjectsWeights = []

  def __init__(self):
    self.Error = False
    self.ErrorException = ''
    self.Test = Result.Data()
    self.Learn = Result.Data()
    
def ReadCommands(commandsFile):
  input = open(commandsFile)
  commands = []
  for line in input:
    command = line.strip()
    if len(command) > 0:
      commands.append(command)
  return commands

def GetProblem(problemName, options):
  if not os.path.exists(options.datasets):
    os.mkdir(options.datasets)
  path = os.path.join(options.datasets, problemName + '.arff')
  if os.path.exists(path):
    return path
  else:
    logger.debug("Requesting problem " + problemName)
    problem = poligon.get_problem(problemName, options.algsynonim, options.algpassword)
    if not problem:
      raise Exception("Could not get problem " + problemName)
    _save_as_arff(problem, problemName, path)
    return path
    
def Substitute(command, prefix, dataFile):
  command = command.replace("%LEARN_INDEXES%", prefix + LEARN_INDEXES_FILE)
  command = command.replace("%TEST_INDEXES%", prefix + TEST_INDEXES_FILE)
  command = command.replace("%PARAMETERS%", prefix + PARAMETERS_FILE)
  #command = command.replace("%PENALTIES%", prefix + PENALTIES_FILE)
  command = command.replace("%LEARN_TARGETS%", prefix + LEARN_TARGETS_FILE)
  command = command.replace("%LEARN_PROBABILITY_MATRIX%", prefix + LEARN_PROBABILITY_MATRIX_FILE)
  command = command.replace("%LEARN_OBJECTS_WEIGHTS%", prefix + LEARN_OBJECTS_WEIGHTS_FILE)
  command = command.replace("%LEARN_PROPERTIES_WEIGHTS%", prefix + LEARN_PROPERTIES_WEIGHTS_FILE)
  command = command.replace("%TEST_TARGETS%", prefix + TEST_TARGETS_FILE)
  command = command.replace("%TEST_PROBABILITY_MATRIX%", prefix + TEST_PROBABILITY_MATRIX_FILE)
  command = command.replace("%TEST_OBJECTS_WEIGHTS%", prefix + TEST_OBJECTS_WEIGHTS_FILE)
  command = command.replace("%TEST_PROPERTIES_WEIGHTS%", prefix + TEST_PROPERTIES_WEIGHTS_FILE)
  command = command.replace("%DATA_FILE%", dataFile)
  return command
  
def Execute(command, timeout):
  logger.debug("Executing command: " + command)
  process = subprocess.Popen(command)
  process.poll()
  while process.returncode == None:
    if timeout[0] < 0:
      process.kill()
      logger.warn("Process timed out")
      return False
    time.sleep(0.01)
    timeout[0] -= 0.01 / 60
    process.poll()
  logger.debug("Command completed")
  if process.returncode == 0:
    return True
  else:
    logger.warn("Process exited with error code: " + str(process.returncode))
    return False

def RunCommands(commands, options, prefix, dataFile):
  timeout = [options.timeout]
  for command in commands:
    command = Substitute(command, prefix, dataFile)
    if not Execute(command, timeout):
      return False
  return True

def GetResult(prefix):
  result = Result()
  result.Learn.Targets = _load_vector(prefix + LEARN_TARGETS_FILE, int)
  result.Learn.ProbabilityMatrix = _load_matrix(prefix + LEARN_PROBABILITY_MATRIX_FILE, float)
  result.Learn.ObjectsWeights = _load_vector(prefix + LEARN_OBJECTS_WEIGHTS_FILE, float)
  result.Learn.PropertiesWeights = _load_vector(prefix + LEARN_PROPERTIES_WEIGHTS_FILE, float)
  result.Test.Targets = _load_vector(prefix + TEST_TARGETS_FILE, int)
  result.Test.ProbabilityMatrix = _load_matrix(prefix + TEST_PROBABILITY_MATRIX_FILE, float)
  result.Test.ObjectsWeights = _load_vector(prefix + TEST_OBJECTS_WEIGHTS_FILE, float)
  result.Test.PropertiesWeights = _load_vector(prefix + TEST_PROPERTIES_WEIGHTS_FILE, float)
  return result
  
def DeleteFile(filename):
  if os.path.exists(filename):
    os.remove(filename)
    
def Clean(prefix):
  DeleteFile(prefix + LEARN_INDEXES_FILE)
  DeleteFile(prefix + TEST_INDEXES_FILE)
  DeleteFile(prefix + PARAMETERS_FILE)
  DeleteFile(prefix + LEARN_TARGETS_FILE)
  DeleteFile(prefix + LEARN_PROBABILITY_MATRIX_FILE)
  DeleteFile(prefix + LEARN_OBJECTS_WEIGHTS_FILE)
  DeleteFile(prefix + LEARN_PROPERTIES_WEIGHTS_FILE)
  DeleteFile(prefix + TEST_TARGETS_FILE)
  DeleteFile(prefix + TEST_PROBABILITY_MATRIX_FILE)
  DeleteFile(prefix + TEST_OBJECTS_WEIGHTS_FILE)
  DeleteFile(prefix + TEST_PROPERTIES_WEIGHTS_FILE)
  
def ProcessTask(task, options, commands):
  problem = GetProblem(task.ProblemSynonim, options)
  results = []
  for i in xrange(len(task.LearnIndexes.ArrayOfInt)):
    logger.debug("Starting " + str(task.PocketId) + ":" + str(i))
    prefix = str(task.PocketId) + "_" + str(i) + "_"
    
    _save_vector(task.LearnIndexes.ArrayOfInt[i].Int, prefix + LEARN_INDEXES_FILE)
    _save_vector(task.TestIndexes.ArrayOfInt[i].Int, prefix + TEST_INDEXES_FILE)
    _save_params(task, prefix + PARAMETERS_FILE)
    
    if RunCommands(commands, options, prefix, problem):
      result = GetResult(prefix)
      Clean(prefix)
    else:
      result = Result()
      result.Error = True
      result.Exception = "Process execution error"
    
    results.append(result)
    
  return results  
    
def Run(options):
  commands = ReadCommands(options.commands)
  
  while True:
  
    while True:
      logger.debug("Requsting task...")
      task = poligon.get_task(options.algsynonim, options.algpassword)
      if task:
        logger.info("Got task for " + task.AlgSynonim + ". Problem: " + task.ProblemSynonim +
                    ", PocketId: " + str(task.PocketId))
        results = ProcessTask(task, options, commands)
        logger.debug("Task is processed. Registering results...")
        poligon.register_results(options.algsynonim, options.algpassword, task.PocketId, results)
        logger.info("Task is processed and results have been registered")
      else:
        break
  
    if options.wait == 0:
      break
    else:
      logger.info("No more tasks. Going to sleep...")
      time.sleep(options.wait * 60)
  logger.info("No more tasks. Exiting...")
  
if __name__ == '__main__':
  logger = _init_logger()
  parser = _init_optparser()

  (options, args) = parser.parse_args()
  logger.debug(options)
  
  try:
    Run(options)
  except Exception as ex:
    logger.error(ex)
