"""
import tvm
import ctypes
import numpy
libvta = ctypes.CDLL("./libvta.so", ctypes.RTLD_GLOBAL)
# ref = tvm.get_global_func("device_api.ext_dev")()
func = tvm.runtime.load_module("../../alu.tar")
dev = tvm.ext_dev(0)
A = tvm.nd.array(numpy.ones((1, 64, 1, 16), dtype='int32'), dev)
B = tvm.nd.array(numpy.ones((1, 64, 1, 16), dtype='int32'), dev)
C = tvm.nd.empty((1, 64, 1, 16), 'int8', dev)
func(A, B, C)
"""
import sys
if sys.platform != "linux":
    raise RuntimeError("This program must run on a Pynq device")
from typing import Tuple

import tvm
import numpy

def init_callback():
    sys.path.append('/usr/local/lib/python3.6/dist-packages')
    import pynq

    driver = pynq.xlnk.Xlnk()
    print(driver.cma_stats())

    rails = pynq.get_rails()
    # power = voltage * current
    # Watt = Volt * Ampere
    power_supply_rail = '12V' # Gives power to the whole board
    dram_rail = '1V2'
    fpga_pl_rail = 'INT'
    recorder = pynq.DataRecorder(
        rails[power_supply_rail].power,
        rails[dram_rail].power,
        rails[fpga_pl_rail].power,
    )

    @tvm.register_func("tvm.contrib.vta.init", override=True)
    def program_fpga(file_name: str) -> None:
        path = tvm.get_global_func("tvm.rpc.server.workpath")(file_name)

        driver.xlnk_reset()

        if not pynq.pl.PL.bitfile_name:
            # TODO: how do I reset the PL do download another bitstream?
            pynq.Bitstream(path).download()

    @tvm.register_func("tvm.contrib.vta.start_recording", override=True)
    def start_recording(interval: float) -> None:
        _ = recorder.record(interval)

    @tvm.register_func("tvm.contrib.vta.stop_recording", override=True)
    def stop_recording() -> Tuple:
        recorder.stop()
        # return recorder._data, recorder._columns
        # NOTE: it seems that only tvm.nd.array can be returned from RPC functions
        return tvm.nd.array(numpy.array(recorder._data), tvm.cpu())

    @tvm.register_func("tvm.contrib.vta.reset_recording", override=True)
    def reset_recording() -> None:
        recorder.reset()

    @tvm.register_func("tvm.contrib.vta.mark_recording", override=True)
    def mark_recording() -> None:
        recorder.mark()

# Taken from tvm.exec.rpc_server
import argparse
import logging
from tvm import rpc

def main(args):
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        parsed args from command-line invocation
    """
    if args.tracker:
        url, port = args.tracker.rsplit(":", 1)
        port = int(port)
        tracker_addr = (url, port)
        if not args.key:
            raise RuntimeError("Need key to present type of resource when tracker is available")
    else:
        tracker_addr = None

    server = rpc.Server(
        args.host,
        args.port,
        args.port_end,
        is_proxy=args.through_proxy,
        key=args.key,
        tracker_addr=tracker_addr,
        load_library=args.load_library,
        custom_addr=args.custom_addr,
        silent=args.silent,
        no_fork=not args.fork,
        server_init_callback=init_callback,
    )
    server.proc.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="The host IP address the tracker binds to"
    )
    parser.add_argument("--port", type=int, default=9090, help="The port of the RPC")
    parser.add_argument(
        "--through-proxy",
        dest="through_proxy",
        action="store_true",
        help=(
            "Whether this server provide service through a proxy. If this is true, the host and"
            "port actually is the address of the proxy."
        ),
    )
    parser.add_argument("--port-end", type=int, default=9199, help="The end search port of the RPC")
    parser.add_argument(
        "--tracker",
        type=str,
        help=("The address of RPC tracker in host:port format. " "e.g. (10.77.1.234:9190)"),
    )
    parser.add_argument(
        "--key", type=str, default="", help="The key used to identify the device type in tracker."
    )
    parser.add_argument("--silent", action="store_true", help="Whether run in silent mode.")
    parser.add_argument("--load-library", type=str, help="Additional library to load")
    parser.add_argument(
        "--no-fork",
        dest="fork",
        action="store_false",
        help="Use spawn mode to avoid fork. This option \
                        is able to avoid potential fork problems with Metal, OpenCL \
                        and ROCM compilers.",
    )
    parser.add_argument(
        "--custom-addr", type=str, help="Custom IP Address to Report to RPC Tracker"
    )

    parser.set_defaults(fork=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.fork is False and not args.silent:
        logging.info(
            "If you are running ROCM/Metal, fork will cause "
            "compiler internal error. Try to launch with arg ```--no-fork```"
        )
    main(args)
