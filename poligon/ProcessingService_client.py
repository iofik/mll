##################################################
# file: ProcessingService_client.py
# 
# client stubs generated by "ZSI.generate.wsdl2python.WriteServiceModule"
#     C:\Python26\Scripts\wsdl2py -b http://poligon.machinelearning.ru/processingservice.asmx?WSDL
# 
##################################################

from ProcessingService_types import *
import urlparse, types
from ZSI.TCcompound import ComplexType, Struct
from ZSI import client
from ZSI.schema import GED, GTD
import ZSI
from ZSI.generate.pyclass import pyclass_type

# Locator
class ProcessingServiceLocator:
    ProcessingServiceSoap_address = "http://poligon.machinelearning.ru/processingservice.asmx"
    def getProcessingServiceSoapAddress(self):
        return ProcessingServiceLocator.ProcessingServiceSoap_address
    def getProcessingServiceSoap(self, url=None, **kw):
        return ProcessingServiceSoapSOAP(url or ProcessingServiceLocator.ProcessingServiceSoap_address, **kw)

# Methods
class ProcessingServiceSoapSOAP:
    def __init__(self, url, **kw):
        kw.setdefault("readerclass", None)
        kw.setdefault("writerclass", None)
        # no resource properties
        self.binding = client.Binding(url=url, **kw)
        # no ws-addressing

    # op: GetTask
    def GetTask(self, request, **kw):
        if isinstance(request, GetTaskSoapIn) is False:
            raise TypeError, "%s incorrect request type" % (request.__class__)
        # no input wsaction
        self.binding.Send(None, None, request, soapaction="http://poligon.machinelearning.ru/GetTask", **kw)
        # no output wsaction
        response = self.binding.Receive(GetTaskSoapOut.typecode)
        return response

    # op: GetProblem
    def GetProblem(self, request, **kw):
        if isinstance(request, GetProblemSoapIn) is False:
            raise TypeError, "%s incorrect request type" % (request.__class__)
        # no input wsaction
        self.binding.Send(None, None, request, soapaction="http://poligon.machinelearning.ru/GetProblem", **kw)
        # no output wsaction
        response = self.binding.Receive(GetProblemSoapOut.typecode)
        return response

    # op: RegisterResult
    def RegisterResult(self, request, **kw):
        if isinstance(request, RegisterResultSoapIn) is False:
            raise TypeError, "%s incorrect request type" % (request.__class__)
        # no input wsaction
        self.binding.Send(None, None, request, soapaction="http://poligon.machinelearning.ru/RegisterResult", **kw)
        # no output wsaction
        response = self.binding.Receive(RegisterResultSoapOut.typecode)
        return response

GetTaskSoapIn = GED("http://poligon.machinelearning.ru/", "GetTask").pyclass

GetTaskSoapOut = GED("http://poligon.machinelearning.ru/", "GetTaskResponse").pyclass

GetProblemSoapIn = GED("http://poligon.machinelearning.ru/", "GetProblem").pyclass

GetProblemSoapOut = GED("http://poligon.machinelearning.ru/", "GetProblemResponse").pyclass

RegisterResultSoapIn = GED("http://poligon.machinelearning.ru/", "RegisterResult").pyclass

RegisterResultSoapOut = GED("http://poligon.machinelearning.ru/", "RegisterResultResponse").pyclass
