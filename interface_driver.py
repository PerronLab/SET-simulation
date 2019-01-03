# interface class

import random
import datetime
import math
from math import pi

from master_driver.interfaces import BaseInterface, BaseRegister, BasicRevert
from csv import DictReader
from StringIO import StringIO
import logging

import RPi.GPIO as IO
import time
import ADC3008 as ADC

#Define GPIO Pins for (CS, CLK, MISO & MOSI)
SPICLK  = 2
SPIMISO = 3 # Dout
SPIMOSI = 4 # Din
SPICS   = 17

_log = logging.getLogger(__name__)
type_mapping = {"string": str,
                "int": int,
                "integer": int,
                "float": float,
                "bool": bool,
                "boolean": bool}

class Interface(BasicRevert, BaseInterface):
    def __init__(self, **kwargs):
        super(Interface, self).__init__(**kwargs)

    def configure(self, config_dict, registry_config_str):
        self.parse_config(registry_config_str)

    def get_point(self, channel_no):
        ADC.setup() # initialize communication
        register = self.get_register_by_name(channel_no) # assign channel to a register

        return register.value

    def _set_point(self, point_name, value):
        register = self.get_register_by_name(point_name)
        if register.read_only:
            raise IOError(
                "Trying to write to a point configured read only: " + point_name)

        register.value = register.reg_type(value)
        return register.value

    def _scrape_all(self):
        result = {}
        read_registers = self.get_registers_by_type("byte", True)
        write_registers = self.get_registers_by_type("byte", False)
        for register in read_registers + write_registers:
            result[register.point_name] = register.value

        return result

    def parse_config(self, configDict):
        if configDict is None:
            return


        for regDef in configDict:
            # Skip lines that have no address yet.
            if not regDef['Point Name']:
                continue

            read_only = regDef['Writable'].lower() != 'true'
            point_name = regDef['Volttron Point Name']
            description = regDef.get('Notes', '')
            channel_no = regDef.get['Channel Number']
            units = regDef['Units']
            default_value = regDef.get("Starting Value", 'sin').strip()
            if not default_value:
                default_value = None
            type_name = regDef.get("Type", 'string')
            reg_type = type_mapping.get(type_name, str)

            register_type = FakeRegister if not point_name.startswith('EKG') else EKGregister

            register = register_type(
                read_only,
                point_name,
                units,
                reg_type,
                channel_no,
                default_value=default_value,
                description=description)

            if default_value is not None:
                self.set_default(point_name, register.value)

            self.insert_register(register)