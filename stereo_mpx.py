#!/usr/bin/env python3
#
# Program to create a variety of stereo multiplex waveform files for the Siglent SDG family of
# Arbitrary Waveform Generators. FM stereo multiplexing is described here:
#           https://en.wikipedia.org/wiki/FM_broadcasting#Stereo_FM
#
# The waveforms produced aid in the repair and alignment of legacy stereo receivers.
# Much in the way analog test equipment such as the Leader LSG-231 or Heathkit IG-37 did
# in the 70's and 80's
#
# This program supports:
#    - Independant left and right tone frequencies and amplitudes
#    - Controllable 19KHz pilot tone level
#    - Controllable L-R subcarrier level
#    - Controllable peak-to-peak output level
# This should allow you to diagnose and adjust FM discriminator and stereo decoder problems in vintage FM tuners
#
# In addition, the total number of samples and the fundamental frequency can be specified.
#   - The fundamental frequency is the frequency at which the entire set of samples is repeated by the
#     waveform generator, and to ensure no artifacts due to discontinuities, both audio frequencies as
#     well as the 19KHz pilot (and therefore the 38KHz subcarrier) should be integer multiples of this
#     fundamental frequency. The default is 100Hz and should be adequate for most uses.
#   - The total number of samples is the number of points that are defined in the resulting file.
#     The default of 524288 (2^19) is plenty to avoid aliasing and other Nyquist problems when used
#     with a fundamental freq of 100Hz on the SDG family of generators.
#
# The easiest way to use this program is to generate a set of files specifying various useful test conditions
# such as:
#
#   - Full stereo, different L + R frequencies:
#            stereo_mpx -f stereo.csv
#   - Same as above without a pilot (should result in mono):
#            stereo_mpx -p 0 -f no_pilot.csv
#   - Stereo signal, only L present:
#            stereo_mpx -r 0 -f left.csv
#   - Stereo signal, only R present:
#            stereo_mpx -l 0 -f right.csv
#
# - Copy these files to a USB stick that the SDG will grok (smaller is better, vfat)
# - Unsert into SDG
# - Select Waveforms->"Arb", "Arb Type" -> "Stored Waveforms", "USB Device (0:)" -> <one of the waveforms above>
#   At this point is should say "Converting ..." and then "File import complete."
#   Repeat for all the waveforms (or just stop there and simply enable the channel output if that is all you need.)
#   Once imported, the waveforms should appear as .bin files with the same name on the internal storage of the
#   SDG, no need to reimport them.
#
#   You can also import the files via EasyWaveX, if you wish.
#
#   To use the signals, you can either directly modulate an RF signal generator that can cover the FM broadcast band,
#   or you can use one channel of the SDG to modulate the other and generate and IF signal suitable for direct injection
#   into a receiver (usually at 10.7 MHz.) Set the modulation to FM with a deviation of 75KHz. You will need to know the
#   correct peak-to-peak level to achieve 75KHz modulation, for the SDG10xxX when using one channel to modulate the
#   other via the Aux In connector and selecting "Source" == "External", this is 12V peak to peak. I believe the correct
#   value is 2V peak-to-peak for a coupole of my old HP RF signal generators. Your equipment will likely vary.
#   Note attempts to directly FM modulate a carrier with an arbitrary waveform on my SDG results in GUI hangs,
#   no idea why. YMMV.
#
# This project was inspired by https://github.com/AI5GW/SIGLENT, which sadly I could not get to work with my SDG10xxX
#
#
# Tested with:
#    - SDG1032X
#    - SDG1060X
#
# TODO - Add PyVISA to directly load into instrument (simialr to the AI5GW project mentioned above)
# TODO - Add pre-emphasis (50 and 75 us time constants & only for audio tones above 3183 and 2122 Hz, respectively) 
# TODO - range check audio frequencies and levels for sanity.
# TODO - check args.fundamental is a divisor of pilot (and therefore subcarrier) freqs
# TODO - check args.{left,right}_frequencies are a multiple of the fundamental
#

# MIT License
# 
# Copyright (c) 2026 AndyW
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


F_PILOT = 19000
F_SUBCARRIER = (F_PILOT * 2)

def build_parser():
    p = argparse.ArgumentParser("stereo-mpx")

    p.add_argument("--filename", "-f", type=str, default='', help="Specify output filename")
    p.add_argument("--pilot", "-p", type=float, default=0.1, help="Pilot level (relative to 1.0) - default 0.1")
    p.add_argument("--right", "-r", type=float, default=1.0, help="Right channel level (relative to 1.0) - default 1.0")
    p.add_argument("--left", "-l", type=float, default=1.0, help="Left channel level (relative to 1.0) - default 1.0")

    p.add_argument("--fundamental", "-F", type=int, default=100, help="Fundamental frequency - default 100 Hz")
    p.add_argument("--left-frequency", "-L", type=int, default=700, help="Left tone frequency - default 700 Hz")
    p.add_argument("--right-frequency", "-R", type=int, default=1800, help="Right tone frequency - default 1800 Hz")

    p.add_argument("--subcarrier", "-S", type=float, default=1.0, help="Subcarrier level (relative to 1.0) - default 1.0")
    p.add_argument("--amplitude", "-a", type=float, default=1.0, help="Output amplitude (peak to peak) - default 1 V")

    p.add_argument("--samples", "-s", type=int, default=(2**19), help=f"Number of samples - default {2**19}")

    p.add_argument("--plot", "-P", help="Plot waveform to file (requires matplotlib)")
    p.add_argument("--title", "-T", help="Title of plot")

    return p

def main():
    args = build_parser().parse_args()

    # create arrays for all the basic component signals
    left = np.zeros(args.samples)
    right = np.zeros(args.samples)
    pilot = np.zeros(args.samples)
    subcarrier = np.zeros(args.samples)

    # Compute the component signals, factoring in their amplitude
    # Make extensive use of the "default=" capability of argparse to provide sensible defaults
    for i in range(args.samples):
        left[i] = np.sin((i * (args.left_frequency / args.fundamental) * 2 * np.pi)/(args.samples - 1)) * args.left
        right[i] = np.sin((i * (args.right_frequency / args.fundamental) * 2 * np.pi)/(args.samples - 1)) * args.right
        pilot[i] = np.sin((i * (F_PILOT / args.fundamental) * 2 * np.pi)/(args.samples - 1)) * args.pilot
        subcarrier[i] = np.sin((i * (F_SUBCARRIER / args.fundamental) * 2 * np.pi)/(args.samples - 1)) * args.subcarrier
    
    # Combine all the basic signals into a composite baseband signal
    baseband = (left + right) + pilot + (subcarrier * (left - right))

    # Brute force determine the peak to peak value of the composite
    # baseband signal, and scale accordingly to obtain the desired peak-to-peak value.
    scale_factor = args.amplitude / (2 * max([abs(max(baseband)), abs(min(baseband))]))
    baseband = scale_factor * baseband

    # Plot using matplotlib, if requested
    if args.plot:
        if MP_AVAILABLE:
            fig, ax = plt.subplots()
            ax.plot(np.arange(args.samples), baseband)
            ax.set_xlabel('Sample')
            ax.set_ylabel('Amplitude (V)')
            ax.set_xlim([0, args.samples])
            if args.title:
                ax.set_title(args.title)
            return plt.savefig(args.plot)
        else:
            sys.stderr.write("Plotting requires matplotlib\n")
            return 1

    # Otherwise, create a csv file of the desired format (brief header, followed by sample data)
    #
    # It appears that siglent needs the CSV file to have DOS line endings,
    # so arrange for that however we are outputting the data - the unix2dos
    # utility is a suitable external converter.
    #
    if args.filename:
        f = open(args.filename, 'w', newline='\r\n')
    else:
        if sys.version_info >= (3, 7):
            f = sys.stdout
            f.reconfigure(newline='\r\n')
        else:
            sys.stderr.write("Externally convert line endings to DOS format\n")
    
    print(f"data length,{args.samples}\nfrequency,{args.fundamental}\namp,{args.amplitude}\noffset,0.0000\nphase,0.0000\n\n\n\n\nxpos,value", file=f)
    
    for i in range(args.samples):
        print(f"{i},{baseband[i]}", file=f)

if __name__ == "__main__":
    main()
