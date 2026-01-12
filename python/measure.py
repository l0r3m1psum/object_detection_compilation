import sys
if sys.platform != "linux":
    raise RuntimeError("This program must run on a Pynq device")

sys.path.append('/usr/local/lib/python3.6/dist-packages')

import pynq
import time
import numpy

A = numpy.ones((2*1024, 1024))
B = numpy.ones((1024, 2*1024))

rails = pynq.get_rails()
# power = voltage * current
# Watt = Volt * Ampere
power_supply_rail = '12V' # Gives power to the whole board
dram_rail = '1V2'
fpga_pl_rail = 'INT'

recorder = pynq.DataRecorder(rails[power_supply_rail].power, rails[dram_rail].power)
with recorder.record(0.02): # Sample every 20 ms
    time.sleep(1)
    C = A@B
    time.sleep(1)
    recorder.mark()
    time.sleep(1)
    C = A@B
    time.sleep(1)

# df = recorder.frame
# (df[power_supply_rail + '_power'] - df[dram_rail + '_power'] ).plot()
