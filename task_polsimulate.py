# PolSimulate - A task to simulate simple full-polarization ALMA data.
#
# Copyright (c) Ivan Marti-Vidal - Nordic ARC Node (2016).
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>,
# or write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# a. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# b. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
# c. Neither the name of the author nor the names of contributors may
#    be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import gc
from simutil import *
import os
import numpy as np
import scipy.interpolate as spint
from taskinit import gentools
from clearcal_cli import clearcal_cli as clearcal
from ft_cli import ft_cli as ft
from simutil import *
ms = gentools(['ms'])[0]
sm = gentools(['sm'])[0]
me = gentools(['me'])[0]
tb = gentools(['tb'])[0]

__version__ = '1.2'

# UNIT TEST LINES:
if __name__ == '__main__':
    vis = '/media/marti/LaCie_3/DATA/NO_BACKUP/polsimulate_output.ms'
    array_configuration = 'alma.out04.cfg'
    incell = ''
    inbright = ''
    incenter = 'J2000 00h00m00.00 -00d00m00.00'
    inwidth = ''

    # TEST 1
    #  impath = '/home/marti/WORKAREA/ARC/DiffPol/EXTENDED_POL/'
    #  images = ['SFits.im','QFits.im','UFits.im']
    #  model_image=[impath + imm for imm in images]
    #  I = [0.2]; Q = [0.03]; U = [0.01]; RM = [1.e7]; spec_index = [0.0];
    #  spectrum_file = '' #/home/marti/WORKAREA/ARC/ARC_TOOLS/PolSim/HOVATTA/model1.txt'
    #  LO=100.e9; BBs = [0.] ; #BBs = [-4.e9,-2.e9,2.e9,4.e9];
    #  spw_width = 2.e9; nchan = 16
    #  corrupt = False

    # TEST 2:
    model_image = []
    I = []
    Q = []
    U = []
    RM = []
    spec_index = []
    spectrum_file = './HOVATTA/model2.txt'
    LO = 233.e9
    BBs = [-9.e9, 9.e9]  # BBs = [-4.e9,-2.e9,2.e9,4.e9]
    spw_width = 1.8e9
    nchan = 16
    #  vis = 'polsimulate_output2.ms'

    Dt_noise = 0.01
    Dt_amp = 0.0
    H0 = -1.5
    onsource_time = 0.25
    observe_time = 3.0
    visib_time = '6s'
    nscan = 5
    t_receiver = 50.0
    tau0 = 0.0
    t_sky = 250.0
    t_ground = 270.0
    seed = 42
    corrupt = False
    feed = 'linear'


def polsimulate(vis='polsimulate_output.ms', array_configuration='alma.out04.cfg', feed='linear',
                LO=100.e9, BBs=[-7.e9, -5.e9, 5.e9, 7.e9], spw_width=2.e9, nchan=128,
                model_image=[], I=[], Q=[], U=[], V=[], RM=[], spec_index=[],
                spectrum_file='',
                incenter='J2000 00h00m00.00 -00d00m00.00', incell='', inbright='',
                inwidth='', H0=-1.5,
                onsource_time=1.5, observe_time=3.0, visib_time='6s', nscan=50,
                corrupt=True, seed=42,
                Dt_amp=0.00, Dt_noise=0.001, tau0=0.0, t_sky=250.0, t_ground=270.0, t_receiver=50.0):

    def printError(msg):
        print '\n', msg, '\n'
        casalog.post('PolSimulate: '+msg)
        raise Exception(msg)

    def printMsg(msg):
        print '\n', msg, '\n'
        casalog.post('PolSimulate: '+msg)

    util = simutil('')

    printMsg('POLSIMULATE - VERSION %s  - Nordic ARC Node' % __version__)

    array = array_configuration[:4].upper()


# ALMA bands:
    Bands = {'3': [84, 119], '5': [163, 211], '6': [211, 275], '7': [
        275, 370], '8': [385, 500], '9': [602, 720], '10': [787, 950]}

# Receiver evector angles. Not used by now:
    Pangs = {'3': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}

    if array == 'ALMA':
        found = False
        for band in Bands.keys():
            if LO/1.e9 > Bands[band][0] and LO/1.e9 < Bands[band][1]:
                found = True
                selb = band
                break

        if not found:
            printError(
                "Frequency %.5fGHz does NOT correspond to any ALMA band!" % (LO/1.e9))
        else:
            printMsg('This is a Band %s ALMA observation.' % selb)

    if feed in ['linear', 'circular']:
        printMsg('Will simulate feeds in %s polarization basis' % feed)
    else:
        printError('Unknown feed %s' % feed)

    # Load the different models:
    # Point source:
    if len(set([len(I), len(Q), len(U), len(V), len(RM), len(spec_index)])) > 1:
        printError("ERROR! I, Q, U, V, RM, and spec_index should all have the same length!")

    # Point source (user-defined spectrum:
    ismodel = False
    if type(spectrum_file) is str and len(spectrum_file) > 0:
        if not os.path.exists(spectrum_file):
            printError("ERROR! spectrum_file is not found!")
        else:
            try:
                ismodel = True
                iff = open(spectrum_file)
                lines = iff.readlines()
                iff.close()
                model = np.zeros((5, len(lines)))
                for li, line in enumerate(lines):
                    temp = map(float, line.split())
                    model[:, li] = temp[:5]
                model2 = model[:, np.argsort(model[0, :])]
                interpI = spint.interp1d(model2[0, :], model2[1, :])
                interpQ = spint.interp1d(model2[0, :], model2[2, :])
                interpU = spint.interp1d(model2[0, :], model2[3, :])
                interpV = spint.interp1d(model2[0, :], model2[4, :])
            except Exception:
                printError("ERROR! spectrum_file has an incorrect format!")

    # Extended source (cube):
    if type(model_image) is list:
        if len(model_image) > 0:
            new_mod = [m + '.polsim' for m in model_image]
            for i in range(4):
                os.system('rm -rf %s' % new_mod[i])
                returnpars = util.modifymodel(model_image[i], new_mod[i],
                                              inbright, incenter, incell,
                                              incenter, inwidth, 0,
                                              flatimage=False)
            Iim, Qim, Uim, Vim = new_mod
        else:
            Iim = ''
            Qim = ''
            Uim = ''
            Vim = ''
    else:
        printError("ERROR! Unkown model_image!")

    if len(Iim) > 0 and not (os.path.exists(Iim) and os.path.exists(Qim)
                             and os.path.exists(Uim) and os.path.exists(Vim)):
        printError("ERROR! one or more model_image components does not exist!")

    if len(model_image) == 0 and len(I) == 0 and not ismodel:
        printError("ERROR! No model specified!")

    antlist = os.getenv("CASAPATH").split(
        ' ')[0] + "/data/alma/simmos/"+array_configuration
    stnx, stny, stnz, stnd, padnames, nant, antnames = util.readantenna(antlist)
    antnames = ["A%02d" % (int(x)) for x in padnames]

    # Setting noise
    if corrupt:
        eta_p, eta_s, eta_b, eta_t, eta_q, t_rx = util.noisetemp(
            telescope=array, freq='%.9fHz' % (LO))
        eta_a = eta_p * eta_s * eta_b * eta_t
        if t_receiver != 0.0:
            t_rx = abs(t_receiver)
        tau0 = abs(tau0)
        t_sky = abs(t_sky)
        t_ground = abs(t_ground)
    else:
        Dt_noise = 0.0
        Dt_amp = 0.0

    os.system('rm -rf '+vis)
    sm.open(vis)

    # Setting the observatory and the observation:
    ALMA = me.observatory(array)
    mount = 'alt-az'
    refdate = '2017/01/01/00:00:00'
    integ = visib_time
    usehourangle = True

    sm.setconfig(telescopename='ALMA', x=stnx, y=stny, z=stnz,
                 dishdiameter=stnd.tolist(),
                 mount=mount, antname=antnames, padname=padnames,
                 coordsystem='global', referencelocation=ALMA)
    spwnames = ['spw%i' % i for i in range(len(BBs))]

    dNu = spw_width/nchan/1.e9
    spwFreqs = []
    dtermsX = []
    dtermsY = []
    ModI = [np.zeros(nchan) for i in BBs]
    ModQ = [np.zeros(nchan) for i in BBs]
    ModU = [np.zeros(nchan) for i in BBs]
    ModV = [np.zeros(nchan) for i in BBs]

    # Spectral windows and D-terms:
    corrp = {'linear': 'XX YY XY YX', 'circular': 'RR LL RL LR'}

    for i in range(len(BBs)):
        Nu0 = (LO+BBs[i]-spw_width/2.)/1.e9
        sm.setspwindow(spwname=spwnames[i], freq='%.8fGHz' % (Nu0),
                       deltafreq='%.9fGHz' % (dNu),
                       freqresolution='%.9fGHz' % (dNu),
                       nchannels=nchan, refcode="BARY",
                       stokes=corrp[feed])
        spwFreqs.append(1.e9*np.linspace(Nu0, Nu0+dNu*nchan, nchan))
        if Dt_amp > 0.0:
            DtX = [[np.random.normal(0., Dt_amp), np.random.normal(
                0., Dt_noise)] for j in stnx]
            DtY = [[np.random.normal(0., Dt_amp), np.random.normal(
                0., Dt_noise)] for j in stnx]
        else:
            DtX = [[0., 0.] for j in stnx]
            DtY = [[0., 0.] for j in stnx]

        if Dt_noise > 0.0:
            dtermsX.append([(np.random.normal(DtX[j][0], Dt_noise, nchan)+1.j
                             * np.random.normal(DtX[j][1], Dt_noise, nchan)) for j in range(len(DtX))])
            dtermsY.append([(np.random.normal(DtY[j][0], Dt_noise, nchan)+1.j
                             * np.random.normal(DtY[j][1], Dt_noise, nchan)) for j in range(len(DtX))])
        else:
            dtermsX.append([np.zeros(nchan, dtype=np.complex128)
                            for j in stnx])
            dtermsY.append([np.zeros(nchan, dtype=np.complex128)
                            for j in stnx])

        # Compute point models:
        if len(I) > 0:
            Lam2 = np.power(299792458./spwFreqs[i], 2.)
            LamLO2 = (299792458./LO)**2.

            for j in range(len(I)):
                p = (Q[j]**2. + U[j]**2.)**0.5 * \
                    np.power(spwFreqs[i]/LO, spec_index[j])
                phi0 = 2.*np.arctan2(U[j], Q[j])
                ModI[i][:] += I[j]*np.power(spwFreqs[i]/LO, spec_index[j])
                ModQ[i][:] += p*np.cos(2.*(RM[j]*(Lam2-LamLO2) + phi0))
                ModU[i][:] += p*np.sin(2.*(RM[j]*(Lam2-LamLO2) + phi0))
                ModV[i][:] += V[j]
        if ismodel:
            ModI[i] += interpI(spwFreqs[i])
            ModQ[i] += interpQ(spwFreqs[i])
            ModU[i] += interpU(spwFreqs[i])
            ModV[i] += interpV(spwFreqs[i])

    # CASA sm tool FAILS with X Y receiver. Will change it later:
    #  sm.setfeed(mode='perfect R L',pol=[''])
    #  sm.setauto(0.0)

    # Field name:
    if len(model_image) > 0:
        source = '.'.join(os.path.basename(model_image[0]).split('.')[:-1])
    else:
        source = 'POLSIM'

    sm.setfield(sourcename=source, sourcedirection=incenter,
                calcode="TARGET", distance='0m')

    mereftime = me.epoch('TAI', refdate)

    printMsg('Will shift the date of observation to match the Hour Angle range')

    sm.settimes(integrationtime=visib_time, usehourangle=usehourangle,
                referencetime=mereftime)

    # Set scans:
    starttimes = []
    stoptimes = []
    sources = []

    scandur = onsource_time/nscan
    T0s = [H0 + (observe_time-scandur)/nscan*i for i in range(nscan)]
    for i in range(nscan):
        sttime = T0s[i]
        endtime = (sttime + scandur)
        starttimes.append(str(3600.*sttime)+'s')
        stoptimes.append(str(3600.*endtime)+'s')
        sources.append(source)

    for n in range(nscan):
        for sp in spwnames:
            sm.observemany(sourcenames=[sources[n]], spwname=sp, starttimes=[
                           starttimes[n]], stoptimes=[stoptimes[n]], project='polsimulate')

    sm.close()

    # Change feeds to XY:
    if feed == 'linear':

        printMsg('CHANGING FEEDS TO X-Y')
        tb.open(vis+'/FEED', nomodify=False)
        pols = tb.getcol('POLARIZATION_TYPE')
        pols[0][:] = 'X'
        pols[1][:] = 'Y'
        tb.putcol('POLARIZATION_TYPE', pols)
        tb.close()

    # Computing par angle:
    ms.open(vis)
    ms.selectinit(datadescid=0)
    temp = ms.getdata(['axis_info', 'ha'], ifraxis=True)
    HAs = temp['axis_info']['time_axis']['HA']/3600.*15.*np.pi/180.
    Ts = temp['axis_info']['time_axis']['MJDseconds']
    ms.close()

    dirst = incenter.split()
    csys = cs.newcoordsys(direction=True)
    csys.setdirection(refcode=dirst[0], refval=' '.join(dirst[1:]))
    Dec = csys.torecord()['direction0']['crval'][1]

    tb.open(vis+'/ANTENNA')
    av = np.average(tb.getcol('POSITION'), axis=1)
    Lat = np.arctan2(av[2], (av[0]**2.+av[1]**2.)**0.5)
    tb.close()

    top = np.sin(HAs)*np.cos(Lat)
    bot = np.sin(Lat)*np.cos(Dec)-np.sin(Dec)*np.cos(HAs)*np.cos(Lat)
    ParAng = np.arctan2(top, bot)

    PAtime = spint.interp1d(Ts, ParAng)

    # Create an auxiliary MS:
    printMsg('Creating the auxiliary single-pol datasets')
    if feed == 'linear':
        polprods = ['XX', 'YY', 'XY', 'YX', 'I', 'Q', 'U', 'V']
    elif feed == 'circular':
        polprods = ['RR', 'LL', 'RL', 'LR', 'I', 'Q', 'U', 'V']

    dvis = [vis + ss for ss in polprods]

    for dv in dvis:
        os.system('rm -rf '+dv)

    sm.open(dvis[0])
    sm.setconfig(telescopename=array, x=stnx, y=stny, z=stnz,
                 dishdiameter=stnd.tolist(),
                 mount=mount, antname=antnames, padname=padnames,
                 coordsystem='global', referencelocation=ALMA)
    spwnames = ['spw%i' % i for i in range(len(BBs))]
    for i in range(len(BBs)):
        sm.setspwindow(spwname=spwnames[i], freq='%.8fGHz' % ((LO+BBs[i]-spw_width/2.)/1.e9),
                       deltafreq='%.9fGHz' % (spw_width/nchan/1.e9),
                       freqresolution='%.9fGHz' % (spw_width/nchan/1.e9),
                       nchannels=nchan, refcode="BARY",
                       stokes=polprods[0])

    #  sm.setfeed(mode='perfect R L',pol=[''])
    sm.setfield(sourcename=source, sourcedirection=incenter,
                calcode="TARGET", distance='0m')

    sm.settimes(integrationtime=visib_time, usehourangle=usehourangle,
                referencetime=mereftime)

    for n in range(nscan):
        for sp in spwnames:
            sm.observemany(sourcenames=[sources[n]], spwname=sp,
                           starttimes=[starttimes[n]], stoptimes=[stoptimes[n]],
                           project='polsimulate')

    sm.close()

    # Simulate Stokes parameters:
    clearcal(vis=dvis[0], addmodel=True)
    clearcal(vis=vis, addmodel=True)

    for dv in dvis[1:]:
        os.system('cp -r %s %s' % (dvis[0], dv))

    # Auxiliary arrays:
    ms.open(dvis[0])
    spwscans = []
    for n in range(len(spwnames)):
        ms.selectinit(datadescid=n)
        spwscans.append(np.copy(ms.range('scan_number')['scan_number']))

    ms.selectinit(datadescid=0)
    ms.select({'scan_number': int(spwscans[0][0])})
    dataI = np.copy(ms.getdata(['data'])['data'])
    dataQ = np.copy(dataI)
    dataU = np.copy(dataI)
    dataV = np.copy(dataI)
    ms.close()

    ntimes = np.shape(dataI)[-1]

    printMsg('Simulating X-Y feed observations')
    PAs = {}
    ant1 = {}
    ant2 = {}

    print "dvis[4]:", dvis[4]
    ms.open(dvis[4], nomodify=False)

    for i in range(len(BBs)):
        print i, spwscans[i], spwscans[i].__class__
        for n in spwscans[i]:
            print "initialize selection %d" % (i)
            ms.selectinit(datadescid=i)
            print "select scan number %d" % (int(n))
            ms.select({'scan_number': int(n)})
            ants = ms.getdata(['antenna1', 'antenna2', 'time'])
            print "ants", ants
            ant1[n] = np.copy(ants['antenna1'])
            print "ant1[n]", ant1[n]
            ant2[n] = np.copy(ants['antenna2'])
            print "ant2[n]", ant2[n]
            MJD = ants['time']
            print "MJD", MJD
            PAs[n] = PAtime(MJD)
            print "PAs[n]", PAs[n]

    ms.close()

    print "Simulating Stokes I"
    if len(Iim) > 0:
        ft(vis=dvis[4], model=Iim, usescratch=True)

    print "Simulating Stokes Q"
    if len(Qim) > 0:
        ft(vis=dvis[5], model=Qim, usescratch=True)

    print "Simulating Stokes U"
    if len(Uim) > 0:
        ft(vis=dvis[6], model=Uim, usescratch=True)

    print "Simulating Stokes V"
    if len(Uim) > 0:
        ft(vis=dvis[7], model=Uim, usescratch=True)

    printMsg('Computing the correlations')
    XX = np.zeros(np.shape(dataI), dtype=np.complex128)
    YY = np.zeros(np.shape(dataI), dtype=np.complex128)
    XY = np.zeros(np.shape(dataI), dtype=np.complex128)
    YX = np.zeros(np.shape(dataI), dtype=np.complex128)

    if corrupt:
        XXa = np.zeros(nchan, dtype=np.complex128)
        YYa = np.zeros(nchan, dtype=np.complex128)
        XYa = np.zeros(nchan, dtype=np.complex128)
        YXa = np.zeros(nchan, dtype=np.complex128)
        XXb = np.zeros(nchan, dtype=np.complex128)
        YYb = np.zeros(nchan, dtype=np.complex128)
        XYb = np.zeros(nchan, dtype=np.complex128)
        YXb = np.zeros(nchan, dtype=np.complex128)

    for i in range(len(BBs)):
        printMsg('Doing spw %i' % i)
        gc.collect()
        for sci, sc in enumerate(spwscans[i]):
            print 'Scan %i of %i' % (sci+1, len(spwscans[i]))
            ms.open(dvis[4], nomodify=False)
            ms.selectinit(datadescid=i)
            ms.select({'scan_number': int(sc)})

            if len(Iim) > 0:
                dataI[:] = ms.getdata(['model_data'])['model_data']
            else:
                dataI[:] = 0.0
            if len(I) > 0 or ismodel:
                dataI[:] += ModI[i][:, np.newaxis]

            ms.close()

            ms.open(dvis[5], nomodify=False)
            ms.selectinit(datadescid=i)
            ms.select({'scan_number': int(sc)})

            if len(Qim) > 0:
                dataQ[:] = ms.getdata(['model_data'])['model_data']
            else:
                dataQ[:] = 0.0
            if len(I) > 0 or ismodel:
                dataQ[:] += ModQ[i][:, np.newaxis]

            ms.close()

            ms.open(dvis[6], nomodify=False)
            ms.selectinit(datadescid=i)
            ms.select({'scan_number': int(sc)})

            if len(Uim) > 0:
                dataU[:] = ms.getdata(['model_data'])['model_data']
            else:
                dataU[:] = 0.0
            if len(I) > 0 or ismodel:
                dataU[:] += ModU[i][:, np.newaxis]

            ms.close()

            ms.open(dvis[7], nomodify=False)
            ms.selectinit(datadescid=i)
            ms.select({'scan_number': int(sc)})

            if len(Uim) > 0:
                dataV[:] = ms.getdata(['model_data'])['model_data']
            else:
                dataV[:] = 0.0
            if len(I) > 0 or ismodel:
                dataV[:] += ModV[i][:, np.newaxis]

            ms.close()

            for j in range(ntimes):
                PA = PAs[sc][j]
                C2 = np.cos(2.*PA)
                S2 = np.sin(2.*PA)
                C = np.cos(PA)
                S = np.sin(PA)
                EPA = np.exp(1.j*2.*PA)
                EMA = np.exp(-1.j*2.*PA)

                if corrupt:

                    # To add a leakage, we have to go to the antenna frame first!

                    # Visibilities in the antenna frame:
                    if feed == 'linear':
                        XXa[:] = (dataI[0, :, j] + dataQ[0, :, j]
                                  * C2 - dataU[0, :, j]*S2)
                        YYa[:] = (dataI[0, :, j] - dataQ[0, :, j]
                                  * C2 + dataU[0, :, j]*S2)
                        XYa[:] = dataU[0, :, j]*C2 + \
                            dataQ[0, :, j]*S2 + 1.j*dataV[0, :, j]
                        YXa[:] = dataU[0, :, j]*C2 + \
                            dataQ[0, :, j]*S2 - 1.j*dataV[0, :, j]

                    if feed == 'circular':
                        XXa[:] = (dataI[0, :, j] + dataV[0, :, j]) * \
                            EPA  # *C2 - dataU[0,:,j]*S2)
                        YYa[:] = (dataI[0, :, j] - dataV[0, :, j]) * \
                            EMA  # + dataU[0,:,j]*S2)
                        XYa[:] = dataQ[0, :, j] + 1.j*dataU[0, :, j]
                        YXa[:] = dataQ[0, :, j] - 1.j*dataU[0, :, j]

                    # Add leakage:
                    XXb[:] = XXa + YYa*dtermsX[i][ant1[sc][j]]*np.conjugate(
                        dtermsX[i][ant2[sc][j]]) + XYa*np.conjugate(dtermsX[i][ant2[sc][j]]) + \
                        YXa*dtermsX[i][ant1[sc][j]]
                    YYb[:] = YYa + XXa*dtermsY[i][ant1[sc][j]]*np.conjugate(
                        dtermsY[i][ant2[sc][j]]) + XYa*dtermsY[i][ant1[sc][j]] + \
                        YXa*np.conjugate(dtermsY[i][ant2[sc][j]])
                    XYb[:] = XYa + YYa*dtermsX[i][ant1[sc][j]] + XXa*np.conjugate(
                        dtermsY[i][ant2[sc][j]]) + YXa*dtermsX[i][ant1[sc][j]]*np.conjugate(dtermsY[i][ant2[sc][j]])
                    YXb[:] = YXa + XXa*dtermsY[i][ant1[sc][j]] + YYa*np.conjugate(
                        dtermsX[i][ant2[sc][j]]) + XYa*dtermsY[i][ant1[sc][j]]*np.conjugate(dtermsX[i][ant2[sc][j]])

                    # Put back into sky frame:
                    if feed == 'linear':
                        XX[0, :, j] = (C*XXb + S*YXb)*C + (C*XYb+S*YYb)*S
                        YY[0, :, j] = -(S*XYb - C*YYb)*C + (S*XXb-C*YXb)*S
                        XY[0, :, j] = (C*XYb + S*YYb)*C - (C*XXb + S*YXb)*S
                        YX[0, :, j] = -(S*XXb - C*YXb)*C - (S*XYb - C*YYb)*S

                    if feed == 'circular':
                        XX[0, :, j] = XXa[:]*EMA
                        YY[0, :, j] = YYa[:]*EPA
                        XY[0, :, j] = XYa[:]
                        YX[0, :, j] = YXa[:]

                else:

                    # No leakage. Compute directly in sky frame:
                    if feed == 'linear':
                        XX[0, :, j] = (dataI[0, :, j] + dataQ[0, :, j])
                        YY[0, :, j] = (dataI[0, :, j] - dataQ[0, :, j])
                        XY[0, :, j] = dataU[0, :, j] + 1.j*dataV[0, :, j]
                        YX[0, :, j] = dataU[0, :, j] - 1.j*dataV[0, :, j]

                    if feed == 'circular':
                        XX[0, :, j] = (dataI[0, :, j] + dataV[0, :, j])
                        YY[0, :, j] = (dataI[0, :, j] - dataV[0, :, j])
                        XY[0, :, j] = dataQ[0, :, j] + 1.j*dataU[0, :, j]
                        YX[0, :, j] = dataQ[0, :, j] - 1.j*dataU[0, :, j]

            # Save:
            print polprods[0]
            ms.open(str(dvis[0]), nomodify=False)
            ms.selectinit(datadescid=i)
            ms.select({'scan_number': int(sc)})
            aux = ms.getdata(['data'])
            aux['data'][:] = XX
            ms.putdata(aux)
            ms.close()
            del aux

            print polprods[1]
            ms.open(str(dvis[1]), nomodify=False)
            ms.selectinit(datadescid=i)
            ms.select({'scan_number': int(sc)})
            aux = ms.getdata(['data'])
            aux['data'][:] = YY[:]
            ms.putdata(aux)
            ms.close()
            del aux

            print polprods[2]
            ms.open(str(dvis[2]), nomodify=False)
            ms.selectinit(datadescid=i)
            ms.select({'scan_number': int(sc)})
            aux = ms.getdata(['data'])
            aux['data'][:] = XY
            ms.putdata(aux)
            ms.close()
            del aux

            print polprods[3]
            ms.open(str(dvis[3]), nomodify=False)
            ms.selectinit(datadescid=i)
            ms.select({'scan_number': int(sc)})
            aux = ms.getdata(['data'])
            aux['data'][:] = YX
            ms.putdata(aux)
            ms.close()
            del aux

    gc.collect()

    # The sm tool IS BROKEN for full-polarization datasets!
    if corrupt:
        printMsg('Corrupting')
        for i in range(len(BBs)):
            printMsg('Doing spw %i' % i)
            for pri, pr in enumerate(dvis[:4]):
                print 'Polprod %s' % polprods[pri]
                sm.openfromms(pr)
                sm.setseed(seed+4*i + 16*pri)
                sm.setdata(fieldid=[sources[0]], spwid=i)
                sm.setnoise(spillefficiency=eta_s, correfficiency=eta_q,
                            antefficiency=eta_a, trx=t_rx,
                            tau=tau0, tatmos=t_sky, tground=t_ground, tcmb=2.725,
                            mode="tsys-manual", senscoeff=-1)
                sm.corrupt()
                sm.done()

    # Copy into full-pol ms:
    printMsg('Saving')

    for i in range(len(BBs)):
        printMsg('Doing spw %i' % i)
        for pri, pr in enumerate(dvis[:4]):
            print 'Polprod %s' % polprods[pri]
            for sc in spwscans[i]:
                ms.open(str(pr), nomodify=False)
                ms.selectinit(datadescid=i)
                ms.select({'scan_number': int(sc)})
                aux = ms.getdata(['data'])['data'][0, :]
                ms.close()
                ms.open(vis, nomodify=False)
                ms.selectinit(datadescid=i)
                ms.select({'scan_number': int(sc)})
                data = ms.getdata(['data'])
                data['data'][pri, :] = aux
                ms.putdata(data)
                ms.close()
                del data, aux

    printMsg('Clearing data')
    del XX, YY, XY, YX, dataI, dataQ, dataU, dataV
    gc.collect()
    clearcal(vis)

    for dv in dvis:
        os.system('rm -rf %s' % dv)

    print 'DONE!\n'

if __name__ == '__main__':
    polsimulate(vis, array_configuration, feed, LO, BBs, spw_width,
                nchan, model_image, I, Q, U, V, RM, spec_index,
                spectrum_file, incenter, incell, inbright, inwidth,
                H0, onsource_time, observe_time, visib_time, nscan,
                corrupt, seed, Dt_amp, Dt_noise, tau0, t_sky,
                t_ground, t_receiver)
